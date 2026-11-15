"""数据管线：分词器（GPT-2 BPE，tiktoken 不可用时退化为字符级）+ bin 缓存 + 批采样。
防泄露约定（细节见 lega/datacheck.py）:
  * 先划分 train/val/test，再做编码等预处理；字符级词表**只在 train 上拟合**，
    val/test 只做 transform，未见字符统一映射 <unk>（BPE 为公开预训练词表，
    不依赖本项目数据，不存在此问题）；
  * 编码前后做硬校验（整数、有限、id 越界）与防泄漏诊断（重复行去污染、
    64-gram 重叠筛查、一元分布漂移），结果写 data_report.json；
  * val/test 内容绝不被改写；测试集只在最终评估时访问（访问账本）。

文档结构约定（缓存协议 v13 起）:
  * token 流**按文档群组分段编码**：token 不跨文档边界，文档边界在 token
    空间因此精确可知；
  * doc_boundaries.json 记录每个 split 的逐文档 token 计数；
  * data_report.structure 给出文档长度分布、跨文档窗口比例（随 ctx 变化）
    与“完全落在单个文档内”的窗口覆盖量。LEGA 的机制主张是“找不到可靠信息
    时少写无关内容”，而跨文档窗口正是数据拼接本身注入的无关内容；不把它
    量化出来，收益就可能来自拼接伪影而不是机制本身。

训练采样口径（受控因子 train_sampler）:
  * ``stream``（默认）：起点在拼接后的 token 流上均匀随机，窗口可以跨文档；
  * ``doc_aligned``：窗口（输入 + 标签）严格落在同一篇文档内，文档权重
    ∝ 文档 token 数（与 stream 保持同一份文档混合比例）。它把“跨文档跳转”
    从隐式拼接伪影变成可单独操控的因子；短文档因装不下窗口而被排除的占比
    必须在报告中可见。

约定 data_dir 内文件：
  train.txt / val.txt     原始文本（prepare_data.py 或 make_dummy_data.py 生成）
  train.bin / val.bin     uint16 token 流（prepare_bins 生成）
  meta.json               {"tokenizer": "gpt2_bpe"|"char", "vocab_size": int, ...}
  data_report.json        防泄漏/清洗/漂移诊断报告（含 source_provenance）
  source_manifest.json    原始 URL/revision/文件大小/SHA-256 来源清单
  doc_boundaries.json     文档边界侧车：每 split 的文档 token 计数（文档边界
                          在 token 空间精确可知，用于跨文档窗口审计与文档内
                          PPL 口径）
"""

import hashlib
import json
import math
import os
import platform
import sys
from typing import List

import numpy as np
import torch

from .datacheck import (
    UNK_TOKEN,
    NEAR_DUP_MIN_CHARS,
    NEAR_DUP_THRESHOLD,
    build_data_report,
    check_vocab_capacity,
    decontaminate_documents,
    decontaminate_near_duplicates,
    decontaminate_texts,
    distribution_shift,
    doc_aligned_windows,
    document_stats as structure_document_stats,
    document_structure_report,
    exact_overlap_alerts,
    grid_window_starts,
    group_disjointness_audit,
    manifest_digest,
    ngram_overlap,
    ngram_overlap_exact,
    post_cleanup_residual,
    read_test_access_log,
    sanitize_text,
    short_ngram_overlap,
    split_documents,
    split_set_digest,
    summarize_test_access,
    text_length_stats,
    token_boundary_positions,
    token_id_range_analysis,
    token_sequence_stats,
    token_stream_stats,
    unknown_char_rate,
    window_boundary_stats,
)
from .datacheck.structure import DOC_STRUCTURE_PROTOCOL, REPORT_CTX_GRID
from .utils import config_bool, config_finite_float, strict_int
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

_GPT2_CACHE = {}

# bin 缓存协议版本：清洗/去污染/词表拟合逻辑变更时必须 +1，
# 旧缓存（含未记录版本的第一代缓存）会被强制重建，
# 否则会用旧协议产出的 bin 静默训练，防泄漏自检报告也与实际数据不符。
# v6：meta 记录 train/val/test.bin 的内容哈希，缓存命中与 load_bin 均核对，
# 防止同尺寸替换 bin 后绕过源文本/报告哈希核对。
# v7：加入近似重复（MinHash/LSH）去污染，并把阈值/min_chars 写入 meta
# 作为缓存身份；旧缓存没有执行该步骤或参数不同，必须重建，否则 data_report
# 会声称已做预处理而实际 bin 仍是未去近重复（或不同阈值）的旧流。
# v8：data_report 新增近重复扫描覆盖率、三档真实删除量、清洗后残留复核与
# 字符/字节核算。旧报告缺少这些闭合证据，若继续命中缓存，汇总页会把“未
# 审计”显示成“无问题”；因此必须重建，让审计字段与 token 流同批产出。
# v9：data_report 记录原始来源溯源（URL/revision/原始文件 SHA-256），meta
# 记录其 digest 作为缓存身份；旧缓存没有“数据究竟来自哪份原始字节”的证据，
# 且原始来源变化而文本恰好未变时旧缓存无法更新报告，必须重建。
# v10：data_report 增加全量 train 对 eval 的 64-gram 精确扫描（含逐字节
# 复核），把采样筛查的“下界”升级为可审计的全量结论；旧缓存没有这份证据，
# 继续命中会让汇总页把“未扫描”当成“无重叠”，必须重建。
# v11：data_report 增加 val/test 内部近似重复簇与块级有效样本量审计；旧
# 缓存缺少这份“评测窗口是否独立”的证据，继续命中会把未审计当成无重复。
# v12：data_report 新增 val/test 内部近似重复簇与块级有效样本量审计；旧
# 缓存缺少这份“评测窗口是否独立”的证据，继续命中会把未审计当成无重复。
# v13：token 流按文档群组分段编码（token 不跨文档边界），新增
# doc_boundaries.json（每 split 的逐文档 token 计数）与 data_report.structure
# （文档长度分布、跨文档窗口比例随 ctx 的变化、文档内窗口覆盖量），并把
# 群组级 split 互斥性审计（群组/段落/行前缀三层）写进 split_protocol。
# 旧缓存既没有边界证据也没有结构审计，继续命中会把“未审计文档结构”当成
# “没有跨文档窗口”，且报告里的群组互斥结论无法复核，必须重建。
CACHE_SCHEMA = 13

# 文档边界侧车文件：记录每个 split 的文档 token 计数（文档边界 = 累计和）。
DOC_BOUNDARIES_FILE = "doc_boundaries.json"


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _source_provenance_digest(extra_report):
    """从上游 extra_report 取来源 manifest 的 digest；无来源证据返回 None。"""
    if not isinstance(extra_report, dict):
        return None
    prov = extra_report.get("source_provenance")
    if not isinstance(prov, dict):
        return None
    recorded = prov.get("digest")
    if isinstance(recorded, str):
        return recorded
    return manifest_digest(prov)


def _bin_hashes_ok(data_dir: str, meta: dict):
    """核对 meta 记录的 bin 内容哈希：True/False/None（旧缓存未记录）。

    只核对大小无法发现“同尺寸替换 train.bin/test.bin”：那样 token 流与
    data_report.json 的去污染/重叠/分布诊断不再对应，训练/评测会静默
    使用被替换的数据。
    """
    recorded = meta.get("bin_sha256")
    if recorded is None:
        return None
    if not isinstance(recorded, dict):
        return False
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}.bin")
        expected = recorded.get(split)
        if expected is None:
            if os.path.exists(path):
                return False
            continue
        if not os.path.exists(path) or _sha256_file(path) != expected:
            return False
    return True


def _cache_integrity(data_dir: str, meta: dict, report_path: str, report_enabled: bool):
    """缓存完整性核对：源文本哈希 + 报告哈希 + bin 内容哈希 + 缓存协议版本。

    返回 (ok, issue, has_sources)。缓存命中只核对大小无法发现“txt 改了但
    bin 没重建”：那样训练集张冠李戴，data_report.json 里的去污染/重叠/
    漂移诊断也不再对应当前数据。只要存在源文本就强校验；只分发 bin（无
    任何 txt）的场景无法核对也无法重建，按不可判定放行，由调用方提示。
    """
    has_sources = any(
        os.path.exists(os.path.join(data_dir, f"{s}.txt"))
        for s in ("train", "val", "test")
    )
    bin_ok = _bin_hashes_ok(data_dir, meta)
    if bin_ok is False:
        return (
            False,
            "bin 内容与构建时记录的哈希不一致"
            "（被替换/篡改/损坏），重建以恢复与源文本一致的 token 流",
            False,
        )
    if not has_sources:
        # 不可判定放行，并把 no_sources 如实返回给调用方（用于提示与缓存判定）
        return True, None, True
    try:
        cache_schema = strict_int(meta.get("cache_schema", 0), "meta.cache_schema")
    except (TypeError, ValueError, OverflowError):
        cache_schema = -1
    if cache_schema != CACHE_SCHEMA:
        return (
            False,
            "缓存由旧版数据管线生成（无版本/源哈希记录），"
            "重建以应用最新清洗与去污染协议",
            False,
        )
    if bin_ok is None:
        return False, "缓存未记录 bin 内容哈希，无法确认 bin 未被替换", False
    recorded = meta.get("source_sha256")
    if not isinstance(recorded, dict):
        return False, "旧缓存未记录源文本哈希，无法确认 bin 与当前 txt 一致", False
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(path):
            continue
        if recorded.get(split) != _sha256_file(path):
            return False, f"{split}.txt 内容与构建 bin 时不一致（缓存过期）", False
    # 文档边界侧车：token 流按文档分段编码，缺了它就无法回答“窗口跨了多少
    # 次文档边界”，报告里的结构审计也会与实际 token 流脱节。
    boundaries_path = os.path.join(data_dir, DOC_BOUNDARIES_FILE)
    boundaries_sha = meta.get("doc_boundaries_sha256")
    if not isinstance(boundaries_sha, str) or not boundaries_sha:
        return False, "缓存未记录文档边界侧车哈希，无法核对结构审计证据", False
    if not os.path.exists(boundaries_path):
        return False, "缺少文档边界侧车 doc_boundaries.json", False
    if _sha256_file(boundaries_path) != boundaries_sha:
        return False, "文档边界侧车与构建 bin 时记录的不一致（被替换/过期）", False
    if report_enabled:
        if not os.path.exists(report_path):
            return False, "缺少防泄漏诊断报告 data_report.json", False
        recorded_report = meta.get("report_sha256")
        if not isinstance(recorded_report, str):
            return False, "旧缓存未记录诊断报告哈希", False
        if _sha256_file(report_path) != recorded_report:
            return False, "诊断报告与构建 bin 时不一致（可能被改写/过期）", False
    return True, None, False


class GPT2Tokenizer:
    """tiktoken gpt2 的轻量包装，统一 encode/decode / max(n_vocab, 1) 接口。"""

    def __init__(self, enc):
        self.enc = enc

    @property
    def name(self) -> str:
        return "gpt2_bpe"

    @property
    def n_vocab(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return list(self.enc.encode(text))

    def encode_batch(self, texts) -> List[List[int]]:
        """批量编码（按文档分段编码时使用）。

        tiktoken 的 ``encode`` 遇到特殊 token 字符串会抛错；批量接口
        ``encode_ordinary_batch`` 会把它们当普通文本静默编码。这里先显式
        扫描并复现同样的拒绝行为，避免脏数据在分段编码路径上被悄悄接受。
        """
        items = list(texts)
        if not items:
            return []
        specials = [t for t in (getattr(self.enc, "special_tokens_set", None) or ())]
        for text in items:
            for special in specials:
                if special and special in text:
                    raise ValueError(
                        f"文本包含 tiktoken 特殊 token {special!r}，拒绝编码"
                    )
        batch = getattr(self.enc, "encode_ordinary_batch", None)
        if batch is not None:
            return [list(ids) for ids in batch(items)]
        return [self.encode(text) for text in items]

    def decode(self, ids) -> str:
        return self.enc.decode([int(i) for i in ids])


class CharTokenizer:
    """字符级分词（离线兜底）。

    add_unk=True 时把 <unk> 固定在 id 0：词表只在 train 上拟合，val/test 中的
    未见字符显式映射到 <unk>，而不是静默混入 0 号真实字符（旧缓存以
    add_unk=False 重建，保持与旧 bin 的编码一致）。
    """

    def __init__(self, itos: List[str], add_unk: bool = False):
        itos = list(itos)
        if add_unk:
            # <unk> 必须唯一且固定在 0；输入词表里可能已带旧格式的
            # <unk>（且不一定在首位），不能简单前置而留下重复项——
            # 否则 stoi 会指向后一个 id，unk_id 不再是 0。
            itos = [UNK_TOKEN] + [ch for ch in itos if ch != UNK_TOKEN]
        self.itos = itos
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.unk_id = self.stoi.get(UNK_TOKEN, 0)

    @property
    def name(self) -> str:
        return "char"

    @property
    def n_vocab(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def encode_batch(self, texts) -> List[List[int]]:
        """批量编码：字符级下每个字符独立，批量与逐段结果完全一致。"""
        return [self.encode(text) for text in texts]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def get_gpt2_tokenizer() -> GPT2Tokenizer:
    if "tok" not in _GPT2_CACHE:
        import tiktoken

        _GPT2_CACHE["tok"] = GPT2Tokenizer(tiktoken.get_encoding("gpt2"))
    return _GPT2_CACHE["tok"]


def get_tokenizer(kind: str, train_text: str = "", val_text: str = ""):
    """kind ∈ {gpt2, auto, char}；gpt2 失败时 auto 退化为字符级，gpt2 直接抛错。

    未知 kind（拼写错误、None、非字符串）必须显式拒绝：静默退化成字符级会
    让产物 meta 记的“请求分词方式”与实际 token 流不一致，后续缓存命中/
    复用判断也随之失真。

    字符级词表只在 train_text 上拟合（val_text 仅为兼容旧调用保留、不参与），
    这保证 val/test 不通过词表构造向训练侧泄露统计信息。
    """
    if not isinstance(kind, str) or kind not in ("gpt2", "auto", "char"):
        raise ValueError(
            f"tokenizer kind 必须是 'gpt2'/'auto'/'char' 之一，收到 {kind!r}"
        )
    if kind in ("gpt2", "auto"):
        try:
            return get_gpt2_tokenizer()
        except Exception as e:  # 无网络且无本地缓存
            if kind == "gpt2":
                raise
            print(f"[data] tiktoken 不可用({e})，退化为字符级分词")
    chars = sorted(set(train_text))
    if not chars:
        chars = list(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,\n"
        )
    return CharTokenizer(chars, add_unk=True)


# ---------------------------------------------------------------- bin 缓存


def tokenizer_fit_source_ok(tokenizer, fit_on) -> bool:
    """分词器种类与 fit_on 声明必须匹配。

    char 词表只能在 train 上拟合；GPT-2 BPE 是公开预训练词表。只检查
    fit_on 是否属于 {train, gpt2_pretrained} 会让 ``tokenizer="char"`` +
    ``fit_on="gpt2_pretrained"``（或反向）通过，等于允许一份被篡改/旧流程
    生成的缓存冒充“安全来源”。tokenizer 缺失时（旧 meta）保持兼容，只按
    fit_on 判断。
    """
    if tokenizer in (None, ""):
        return fit_on in ("train", "gpt2_pretrained")
    return (str(tokenizer) == "char" and fit_on == "train") or (
        str(tokenizer) == "gpt2_bpe" and fit_on == "gpt2_pretrained"
    )


def prepare_bins(
    data_dir: str,
    kind: str = "auto",
    force: bool = False,
    report: bool = True,
    extra_report: dict = None,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
    near_dup_min_chars: int = NEAR_DUP_MIN_CHARS,
) -> dict:
    """把 train.txt/val.txt（以及若存在的 test.txt）编码为 uint16 bin
    （已存在则跳过，force=True 重建）。

    这是所有文本进入训练/评测的唯一入口，因此在此集中执行防泄漏处理：
      1) 文本清洗（坏字符）、跨 split 精确去污染与近似重复去污染（只改
         train；近似去污染用 near_dup_threshold/near_dup_min_chars 作为
         协议参数，默认 0.80/200 字符）；
      2) 分词器只在 train 上拟合（字符级词表 + <unk>）；
      3) val/test 只 transform，随后对三条 token 流做硬校验；
      4) 生成 data_report.json（重复、近似重复、采样与全量精确 64-gram
         重叠、分布漂移、访问账本）。
    extra_report 用于记录数据来源/划分协议等上游元数据（原样并入报告）。
    """
    # 近似去污染参数既是本次构建口径，也是缓存身份。必须在读缓存前严格
    # 校验：否则 bool/小数参数可能在 decontaminate_near_duplicates 校验前
    # 就命中旧缓存，绕过校验并把不同口径的 bin 静默复用。
    if isinstance(near_dup_threshold, (bool, np.bool_)):
        raise ValueError(
            f"near_dup_threshold 必须是 (0,1] 内的数值，收到 {near_dup_threshold!r}"
        )
    try:
        near_dup_threshold = config_finite_float(
            near_dup_threshold, "near_dup_threshold"
        )
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            f"near_dup_threshold 必须是 (0,1] 内的数值，收到 {near_dup_threshold!r}"
        ) from e
    if not 0.0 < near_dup_threshold <= 1.0:
        raise ValueError(
            f"near_dup_threshold 必须在 (0,1] 内，收到 {near_dup_threshold!r}"
        )
    near_dup_min_chars = strict_int(near_dup_min_chars, "near_dup_min_chars")
    if near_dup_min_chars < 1:
        raise ValueError(f"near_dup_min_chars 必须 >= 1，收到 {near_dup_min_chars!r}")
    meta_path = os.path.join(data_dir, "meta.json")
    report_path = os.path.join(data_dir, "data_report.json")
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    test_txt_path = os.path.join(data_dir, "test.txt")
    has_test_txt = os.path.exists(test_txt_path)
    test_bin = os.path.join(data_dir, "test.bin")
    if (
        os.path.exists(meta_path)
        and os.path.exists(train_bin)
        and os.path.exists(val_bin)
        and not force
    ):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = None
        # 语法合法但结构畸形（数组/标量）的缓存元数据按“无 meta”处理，
        # 走重建路径，而不是在 .get 上崩溃（与指纹缓存同口径）。
        if not isinstance(meta, dict):
            meta = {}
        # test.txt 存在而 test.bin 缺失/不一致时缓存不完整，必须重建，
        # 否则评测会静默回退到 val（与 checkpoint 选择同一 split）。
        # 缓存的分词器与本次请求的 kind 不一致时必须重建，
        # 否则会静默沿用旧分词方式产生的 bin。
        cached_kind = meta.get("requested_kind")
        if cached_kind is None:
            # 旧版 meta 无 requested_kind：按实际分词器推断，
            # 避免 char 缓存被 gpt2 请求静默复用（反之亦然）。
            cached_kind = "gpt2" if meta.get("tokenizer") == "gpt2_bpe" else "char"
        # 近似去污染参数是缓存身份的一部分：同一份 txt 用不同阈值构建出的
        # bin 不同，不能因为源文件哈希未变就静默复用旧阈值的 token 流。
        cached_near = meta.get("near_dup")
        near_ok = False
        if isinstance(cached_near, dict):
            try:
                cached_threshold = config_finite_float(
                    cached_near.get("threshold"), "meta.near_dup.threshold"
                )
                cached_min_chars = strict_int(
                    cached_near.get("min_chars"), "meta.near_dup.min_chars"
                )
                near_ok = (
                    cached_threshold == near_dup_threshold
                    and cached_min_chars == near_dup_min_chars
                )
            except (TypeError, ValueError, OverflowError):
                near_ok = False
        # 旧缓存可能是在 val/test 上拟合词表（泄露）或缺少防泄漏报告，必须重建
        fit_ok = tokenizer_fit_source_ok(meta.get("tokenizer"), meta.get("fit_on"))
        integrity_ok, integrity_issue, no_sources = _cache_integrity(
            data_dir, meta, report_path, report
        )
        # 原始来源身份：仅当本次调用显式携带 provenance 时核对。没有来源
        # 证据的旧调用（例如只对已有 txt 直接 prepare_bins）不应被误判。
        expected_prov_digest = _source_provenance_digest(extra_report)
        prov_ok = (
            expected_prov_digest is None
            or meta.get("source_provenance_digest") == expected_prov_digest
        )
        if no_sources:
            # 纯 bin 目录无法重建，也无法核对近似去污染参数/源哈希；与
            # _cache_integrity 的“不可判定放行”同口径，保留旧行为并让
            # 调用方打印警告，而不是因缺少 train.txt 在重建路径崩溃。
            near_ok = True

        def _meta_int(key):
            # 缓存是派生数据，字段缺失/类型异常按“与 bin 不一致”处理。
            # bool 是 int 子类，小数 int() 会截断；两者都能伪造出与 bin
            # 大小相符的 token 数，让损坏/过期缓存被当成有效。
            raw = meta.get(key, -1)
            if isinstance(raw, (bool, np.bool_)):
                return -1
            if isinstance(raw, str) and "_" in raw:
                return -1
            try:
                f = float(raw)
            except (TypeError, ValueError, OverflowError):
                return -1
            if not math.isfinite(f) or f < 0 or not f.is_integer():
                return -1
            try:
                return int(f)
            except (OverflowError, ValueError):
                return -1

        train_valid = os.path.getsize(train_bin) == _meta_int("train_tokens") * 2
        val_valid = os.path.getsize(val_bin) == _meta_int("val_tokens") * 2
        # 无源 txt（纯 bin 数据目录）时无法判定 test.bin 是否对应缺失的
        # test.txt：显式保留 test.bin 视为有效，否则该场景会在重建路径上
        # 因缺少 train.txt 直接报错。
        test_valid = (
            not has_test_txt and (no_sources or not os.path.exists(test_bin))
        ) or (
            has_test_txt
            and os.path.exists(test_bin)
            and os.path.getsize(test_bin) == _meta_int("test_tokens") * 2
        )
        if (
            (cached_kind == kind or kind == "auto")
            and train_valid
            and val_valid
            and test_valid
            and fit_ok
            and near_ok
            and integrity_ok
            and prov_ok
        ):
            if no_sources:
                print(
                    "[data] 警告：数据目录只有 bin 没有源 txt，"
                    "无法核对缓存与源文本一致性（按不可判定放行）"
                )
            if kind == "auto":
                # auto 的实际分词器取决于当前环境（tiktoken 是否可用），
                # 直接复用缓存可能与本应使用的分词器不一致，必须提示。
                try:
                    now_tok = get_gpt2_tokenizer().name
                except Exception:
                    now_tok = "char"
                if meta.get("tokenizer") != now_tok:
                    print(
                        f"[data] 警告：kind=auto 复用了 {meta.get('tokenizer')} 缓存，"
                        f"但当前环境 auto 将使用 {now_tok}；如需一致请 force=True 重建"
                    )
            return meta
        if not (train_valid and val_valid and test_valid):
            print("[data] 缓存 bin/meta 大小不一致，重建 bin")
        elif not fit_ok:
            print("[data] 旧缓存词表可能在全量数据上拟合（泄露风险），重建 bin")
        elif not near_ok:
            print("[data] 近似去污染参数与缓存不一致（或旧缓存未记录），重建 bin")
        elif not integrity_ok:
            print(f"[data] {integrity_issue}，重建 bin")
        elif not prov_ok:
            print(
                "[data] 原始来源 provenance 与缓存身份不一致（来源已更新），"
                "重建 bin 以更新报告"
            )
        else:
            print(f"[data] 缓存分词方式({cached_kind})与请求({kind})不一致，重建 bin")

    texts = {}
    with open(os.path.join(data_dir, "train.txt"), "r", encoding="utf-8") as f:
        texts["train"] = f.read()
    with open(os.path.join(data_dir, "val.txt"), "r", encoding="utf-8") as f:
        texts["val"] = f.read()
    if has_test_txt:
        with open(test_txt_path, "r", encoding="utf-8") as f:
            texts["test"] = f.read()

    # 源文本指纹：缓存命中时据此确认 bin 确实由当前 train/val/test.txt 构建，
    # 防止“txt 改了但 bins 没重建”的静默过期（防泄漏报告也会随之失真）。
    # 同时算出集合级指纹：逐 split 哈希证明“每个文件没变”，集合级指纹证明
    # “三个文件来自同一次划分”（混合版本划分逐文件看都没变）。
    source_sha256 = {}
    for split, path in (
        ("train", os.path.join(data_dir, "train.txt")),
        ("val", os.path.join(data_dir, "val.txt")),
        ("test", test_txt_path),
    ):
        if os.path.exists(path):
            source_sha256[split] = _sha256_file(path)
    split_set_digest_value = split_set_digest(source_sha256)

    # 1) 清洗 + 去污染：坏字符确定性删除/替换；与 val/test 精确重复的长行
    #    只从 train 移除（val/test 绝不改写，保持尘封）。
    sanitization = {}
    for name in list(texts):
        clean, stats = sanitize_text(texts[name])
        texts[name] = clean
        sanitization[name] = stats
    # 块级去污染先于行级：整块复制（可能由多行短行组成）一旦被行级先删掉
    # 其中一条长行，块哈希就不再与 eval 匹配，剩余短行会漏检。
    texts, doc_dup_stats = decontaminate_documents(texts)
    texts, duplicate_stats = decontaminate_texts(texts, min_chars=40)
    # 行级删除会改变块结构：删掉块中的长行后，剩余短行可能恰好重新拼成
    # 与 val/test 完全相同的多行块。只做“先块后行”会漏掉这种新生成的
    # 副本，因此行级确有删除时再跑一遍块级去污染（只会继续删，必然终止）。
    line_dropped = sum(
        duplicate_stats[k]
        for k in (
            "train_lines_dropped_as_duplicates_of_eval",
            "train_lines_dropped_as_normalized_duplicates_of_eval",
            "train_intra_split_duplicate_lines_removed",
            "train_intra_split_normalized_duplicate_lines_removed",
        )
    )
    if line_dropped:
        texts, post_doc_stats = decontaminate_documents(texts)
        for key in (
            "train_blocks_dropped_as_duplicates_of_eval",
            "train_intra_split_duplicate_blocks_removed",
            "train_lines_dropped",
            "train_chars_dropped",
        ):
            doc_dup_stats[key] = int(doc_dup_stats.get(key, 0)) + int(
                post_doc_stats.get(key, 0)
            )
    duplicate_stats["document_level"] = doc_dup_stats
    # 近似重复去污染：精确行/块去重只能抓住逐字节（或仅空白/NFKC 差异）
    # 的复制；改了几个词的副本仍会让 train 记住 val/test 的内容、并在 train
    # 内部放大同一段文本的权重。阈值是协议参数（0.80），报告同时给出
    # 0.70/0.80/0.90 三档敏感性与 LSH 理论召回率，删除量对阈值的依赖可见。
    texts, near_stats = decontaminate_near_duplicates(
        texts, threshold=near_dup_threshold, min_chars=near_dup_min_chars
    )
    duplicate_stats["near_duplicate"] = near_stats
    if near_stats["train_blocks_dropped_total"]:
        print(
            f"[data] 近似重复去污染：从 train 移除 "
            f"{near_stats['train_blocks_dropped_as_near_duplicates_of_eval']:,} "
            f"个与 val/test 近似的块、"
            f"{near_stats['train_intra_split_near_duplicates_removed']:,} 个 "
            f"train 内部近似重复块（共 {near_stats['train_lines_dropped']:,} 行；"
            "val/test 不改写）"
        )
        # 删除整块可能让相邻块合并成新的精确副本；再跑一次精确块去重到
        # 不动点。结果单列，避免把“近似去重后的补充清理”混进首轮统计。
        texts, post_doc_stats = decontaminate_documents(texts)
        for key in (
            "train_blocks_dropped_as_duplicates_of_eval",
            "train_intra_split_duplicate_blocks_removed",
            "train_lines_dropped",
            "train_chars_dropped",
        ):
            doc_dup_stats[key] = int(doc_dup_stats.get(key, 0)) + int(
                post_doc_stats.get(key, 0)
            )
        duplicate_stats["post_near_dup_exact_cleanup"] = post_doc_stats
    # 清洗后复核：不只报告“删了多少”，还要证明最终 train 里没有残留的
    # 精确/归一化跨 split 长行或多行块。该核对不改写任何 split，失败只
    # 触发告警（供下游硬闸门/人工复核），不会静默放过。
    duplicate_stats["post_cleanup"] = post_cleanup_residual(
        texts, min_chars=40, min_lines=2
    )
    # 近似去污染是自动改写训练集，必须防止“阈值/语料异常导致 train 被删空”
    # 被当成一次正常实验静默跑完：>90% 或删空直接失败，要求人工核查。
    if near_stats["removal_fraction_chars"] > 0.90:
        raise ValueError(
            "近似重复去污染将移除 train 可用字符的 "
            f"{near_stats['removal_fraction_chars']:.1%}（>90%）：语料可能"
            "高度重复或阈值过宽，拒绝静默产出几乎为空的训练集；请核查 "
            "data_report 的 duplicates.near_duplicate 或调整协议阈值"
        )
    if near_stats["train_blocks_dropped_total"] and not texts["train"].strip():
        raise ValueError(
            "近似重复去污染后 train 为空：所有训练块都被判为 val/test 的"
            "近似副本；请核查去污染证据后再决定是否调整阈值/划分"
        )
    train_text, val_text, test_text = texts["train"], texts["val"], texts.get("test")
    # 原文此后只由局部变量引用：逐段编码完即释放
    del texts
    n_doc_cross = doc_dup_stats["train_blocks_dropped_as_duplicates_of_eval"]
    n_doc_intra = doc_dup_stats["train_intra_split_duplicate_blocks_removed"]
    if n_doc_cross or n_doc_intra:
        print(
            f"[data] 块级去污染：从 train 移除 {n_doc_cross:,} 个与 val/test "
            f"重复的多行块、{n_doc_intra:,} 个内部重复块"
            f"（共 {doc_dup_stats['train_lines_dropped']:,} 行；"
            "val/test 不改写）"
        )
    if doc_dup_stats["val_test_overlap_blocks"]:
        print(
            f"[data] 警告：val 与 test 有 "
            f"{doc_dup_stats['val_test_overlap_blocks']:,} 个多行块重复"
            "（两侧均未改写；val 选型对 test 的独立性下降）"
        )
    n_cross = (
        duplicate_stats["train_lines_dropped_as_duplicates_of_eval"]
        + duplicate_stats["train_lines_dropped_as_normalized_duplicates_of_eval"]
    )
    n_intra = (
        duplicate_stats["train_intra_split_duplicate_lines_removed"]
        + duplicate_stats["train_intra_split_normalized_duplicate_lines_removed"]
    )
    if n_cross:
        print(
            f"[data] 去污染：从 train 移除 {n_cross:,} 条与 val/test "
            "重复（精确/空白归一化）的长行"
        )
    if n_intra:
        print(
            f"[data] 去重：从 train 移除 {n_intra:,} 条内部重复长行"
            "（保留首次出现；val/test 不改写）"
        )
    if duplicate_stats["val_test_overlap_long_lines"]:
        print(
            f"[data] 警告：val 与 test 有 "
            f"{duplicate_stats['val_test_overlap_long_lines']:,} 条长行重复"
            "（两侧均未改写；val 选型对 test 的独立性下降）"
        )

    # 2) 分词器只在 train 上拟合；3) val/test 只做 transform
    tok = get_tokenizer(kind, train_text)
    check_vocab_capacity(tok.n_vocab)
    print(f"[data] 分词器: {tok.name}（词表仅在 train 上拟合）, vocab={tok.n_vocab}")

    unk_rates = {}
    for name, txt in (("train", train_text), ("val", val_text), ("test", test_text)):
        if txt is not None:
            unk_rates[name] = unknown_char_rate(tok, txt)
    if unk_rates.get("test") is not None and unk_rates["test"] > 0.001:
        print(
            f"[data] 提示：test 字符级 OOV 率 {unk_rates['test']:.4%}，"
            "未见字符映射 <unk>（词表只用 train 拟合，防止泄露）"
        )

    # 文档结构：先切分再编码，token 不跨文档边界，文档边界在 token 空间
    # 精确可知（“跨文档窗口比例”因此是可计算的事实而不是近似）。
    split_texts = {"train": train_text, "val": val_text}
    if test_text is not None:
        split_texts["test"] = test_text
    doc_segments = {}
    doc_group_unit = {}
    for name, txt in split_texts.items():
        info = split_documents(txt)
        doc_segments[name] = info["segments"]
        doc_group_unit[name] = info["group_unit"]

    # 群组级 split 互斥性审计：把“按文档划分”的声明升级为可复核结论。
    # 整篇文档/段落同时出现在两个 split 是文档被复制的直接证据；行前缀
    # 共享提示文档被从行中间切开（逐行比较看不到），只作告警并给样例。
    group_audit = group_disjointness_audit(split_texts, min_chars=40)
    if group_audit["problems"]:
        print(
            f"[data] ⚠ 群组级互斥性审计发现 {group_audit['verdict']}："
            f"{group_audit['pairs']}"
        )
    else:
        print("[data] 群组级互斥性审计：三个 split 之间无共享文档/段落/行前缀")
    # 群组互斥性结论并入 split_protocol：它是“按文档划分”声明的可复核证据，
    # 放在 split_protocol 下才能被预检与汇总页按协议违规口径消费。必须在
    # build_data_report 之前合并——报告生成时会据此产出协议告警。
    _extra = extra_report if isinstance(extra_report, dict) else {}
    _split_protocol = _extra.get("split_protocol")
    _split_protocol = dict(_split_protocol) if isinstance(_split_protocol, dict) else {}
    _split_protocol["group_disjointness"] = group_audit
    _extra = dict(_extra)
    _extra["split_protocol"] = _split_protocol
    extra_report = _extra

    def encode_stream(text: str) -> np.ndarray:
        parts: List[np.ndarray] = []
        # 分块编码；块边界对齐到换行符，
        # 使跨块 BPE 合并的偏差最小（GPT-2 BPE 极少跨行合并）。
        CHUNK = 5_000_000
        pos, n = 0, len(text)
        while pos < n:
            end = min(pos + CHUNK, n)
            if end < n:
                nl = text.rfind("\n", pos + 1, end)
                if nl > pos:
                    end = nl + 1
            e = tok.encode(text[pos:end])
            arr = np.asarray(e)
            if arr.size:
                if arr.dtype.kind not in "iu":
                    raise ValueError(
                        f"分词器返回非整数 token（dtype={arr.dtype}），"
                        f"拒绝编码 {pos}:{end}"
                    )
                wide = arr.astype(np.int64)
                lo, hi = int(wide.min()), int(wide.max())
                if lo < 0 or hi > 65535:
                    # 先 astype(uint16) 再校验会让 70000 环绕成 4464，
                    # token_stream_stats 只看到合法 id，脏数据被静默接受。
                    raise ValueError(
                        f"token id 越界 uint16（min={lo}, max={hi}，"
                        f"分块 {pos}:{end}）：拒绝静默环绕，请检查分词器/词表"
                    )
                parts.append(wide.astype(np.uint16))
            else:
                parts.append(np.zeros(0, dtype=np.uint16))
            pos = end
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)

    def encode_documents(name: str, text: str):
        """按文档群组分段编码，返回 (token 流, 逐文档 token 数, 结构审计)。

        分段编码保证没有 token 跨过文档边界；字符级分词器下每个字符独立，
        逐文档 token 数等于字符数，因此直接按整段流切分即可，无需重复编码。
        """
        segments = doc_segments[name]
        group_unit = doc_group_unit[name]
        if not segments:
            return (
                np.zeros(0, dtype=np.uint16),
                [],
                document_structure_report(text, segments, group_unit, []),
            )
        if tok.name == "char":
            stream = encode_stream(text)
            counts = [int(e) - int(s) for s, e in segments]
            if int(sum(counts)) != int(stream.size):
                raise ValueError(
                    f"{name} 的逐文档 token 数之和（{sum(counts)}）与 token 流长度"
                    f"（{stream.size}）不一致：文档分段未覆盖全文，拒绝落盘"
                )
            return (
                stream,
                counts,
                document_structure_report(text, segments, group_unit, counts),
            )
        pieces = tok.encode_batch([text[int(s) : int(e)] for s, e in segments])
        parts = []
        counts = []
        for (s, e), ids in zip(segments, pieces):
            arr = np.asarray(ids)
            if arr.size and arr.dtype.kind not in "iu":
                raise ValueError(
                    f"分词器返回非整数 token（dtype={arr.dtype}），"
                    f"拒绝编码 {name} 文档 {s}:{e}"
                )
            wide = arr.astype(np.int64) if arr.size else arr
            if wide.size:
                lo, hi = int(wide.min()), int(wide.max())
                if lo < 0 or hi > 65535:
                    raise ValueError(
                        f"token id 越界 uint16（min={lo}, max={hi}，"
                        f"{name} 文档 {s}:{e}）：拒绝静默环绕，请检查分词器/词表"
                    )
            counts.append(int(wide.size))
            parts.append(
                wide.astype(np.uint16) if wide.size else np.zeros(0, np.uint16)
            )
        stream = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)
        return (
            stream,
            counts,
            document_structure_report(text, segments, group_unit, counts),
        )

    tr, tr_doc_counts, tr_structure = encode_documents("train", train_text)
    text_stats = {"train": text_length_stats(train_text)}
    train_text = None
    va, va_doc_counts, va_structure = encode_documents("val", val_text)
    text_stats["val"] = text_length_stats(val_text)
    val_text = None
    te = None
    te_doc_counts = []
    te_structure = None
    if test_text is not None:
        te, te_doc_counts, te_structure = encode_documents("test", test_text)
        text_stats["test"] = text_length_stats(test_text)
        test_text = None
    doc_token_counts = {"train": tr_doc_counts, "val": va_doc_counts}
    structure_splits = {"train": tr_structure, "val": va_structure}
    if te is not None:
        doc_token_counts["test"] = te_doc_counts
        structure_splits["test"] = te_structure
    # 分段编码后原文不再需要：立刻释放，避免大语料常驻内存。
    split_texts.clear()
    del split_texts

    # 4) token 流硬校验 + 防泄漏诊断
    streams = {"train": tr, "val": va}
    if te is not None:
        streams["test"] = te
    stream_stats = {}
    for name, arr in streams.items():
        stats = token_stream_stats(arr, tok.n_vocab, name)
        stream_stats[name] = stats
    # 字符级分词器把所有未见字符映射到同一个 <unk>：不同 split 的 OOV 长串
    # 会在 token 层变成完全相同的 0 序列，若直接做 n-gram 重叠会被误判为
    # “语料复制”并触发 hard fail。排除含 unk 的窗口，只让真实 token 作为
    # 泄漏证据（原文级重复仍由行/块去污染与 verify_splits_disjoint 检查）。
    ignore_ids = None
    if tok.name == "char" and UNK_TOKEN in getattr(tok, "stoi", {}):
        ignore_ids = {int(tok.unk_id)}
    contamination = {"train_val": ngram_overlap(tr, va, ignore_ids=ignore_ids)}
    # 短 n-gram 筛查：64-gram 会漏掉“整句/整段复用但 <64 token”的模板化泄露；
    # 短 n 更易命中自然搭配，因此只作诊断告警（见 leakage_alerts）。
    contamination_short = {
        "train_val": short_ngram_overlap(tr, va, ignore_ids=ignore_ids)
    }
    # 采样 64-gram 重叠只是下界：train/eval 两侧都可能漏抽。这里对 train
    # 全量扫描、eval 默认全窗口检查，命中后再逐字节复核，结论不再依赖随机
    # 采样；OOV 窗口仍按 ignore_ids 排除，避免 <unk> 碰撞被当成复制。
    contamination_exact = {
        "train_val": ngram_overlap_exact(tr, va, ignore_ids=ignore_ids)
    }
    distribution = {"val_vs_train": distribution_shift(tr, va, tok.n_vocab)}
    if te is not None:
        contamination["train_test"] = ngram_overlap(tr, te, ignore_ids=ignore_ids)
        # eval 两侧也要互查：val 用于选 checkpoint，test 用于最终报告；
        # 二者若有共享片段，val 选择会部分“预演”test，test 不再独立。
        contamination["val_test"] = ngram_overlap(va, te, ignore_ids=ignore_ids)
        contamination_short["train_test"] = short_ngram_overlap(
            tr, te, ignore_ids=ignore_ids
        )
        contamination_short["val_test"] = short_ngram_overlap(
            va, te, ignore_ids=ignore_ids
        )
        contamination_exact["train_test"] = ngram_overlap_exact(
            tr, te, ignore_ids=ignore_ids
        )
        contamination_exact["val_test"] = ngram_overlap_exact(
            va, te, ignore_ids=ignore_ids, reference_split="val"
        )
        distribution["test_vs_train"] = distribution_shift(tr, te, tok.n_vocab)
        distribution["test_vs_val"] = distribution_shift(va, te, tok.n_vocab)
    tokenizer_report = {
        "kind": tok.name,
        "fit_on": "train" if tok.name == "char" else "gpt2_pretrained",
        "vocab_size": int(tok.n_vocab),
        "unk_token": (
            UNK_TOKEN if getattr(tok, "stoi", {}).get(UNK_TOKEN) is not None else None
        ),
        "unk_rate": {k: v for k, v in unk_rates.items() if v is not None},
    }

    # 收集额外的 token 分布诊断指标
    range_analysis = {}
    sequence_stats = {}
    for name, arr in streams.items():
        try:
            ra = token_id_range_analysis(arr, tok.n_vocab, name)
            if ra.get("range_analysis") is not None:
                range_analysis[name] = ra
        except Exception:
            import logging as _logging; _logging.warning("Suppressed exception in data.py:936", exc_info=True)
        try:
            ss = token_sequence_stats(arr)
            if ss.get("tokens", 0) > 0:
                sequence_stats[name] = ss
        except Exception:
            import logging as _logging; _logging.warning("Suppressed exception in data.py:942", exc_info=True)

    data_report = build_data_report(
        tokenizer=tokenizer_report,
        splits=stream_stats,
        sanitization=sanitization,
        duplicates=duplicate_stats,
        contamination={k: v for k, v in contamination.items()},
        distribution={k: v for k, v in distribution.items()},
        text_stats=text_stats,
        extra=extra_report,
    )

    # 额外写入 token 分布诊断
    data_report["token_range_analysis"] = range_analysis
    data_report["token_sequence_stats"] = sequence_stats
    # 划分集合级指纹：证明三个 split 来自同一次划分（逐文件哈希证明不了）。
    data_report["split_set"] = {
        "digest": split_set_digest_value,
        "splits": dict(source_sha256),
    }
    # 文档结构证据（v13）：文档长度分布、跨文档窗口比例随 ctx 的变化、文档内
    # 窗口覆盖量。LEGA 的机制主张是“找不到可靠信息时少写无关内容”，而跨
    # 文档窗口正是数据拼接本身注入的无关内容；不量化它，收益就可能来自
    # 拼接伪影而不是机制本身。
    doc_boundaries_path = os.path.join(data_dir, DOC_BOUNDARIES_FILE)
    doc_boundaries = {
        "protocol": DOC_STRUCTURE_PROTOCOL,
        "group_disjointness_protocol": group_audit.get("protocol"),
        "tokenizer": tok.name,
        "vocab_size": int(tok.n_vocab),
        "encoding_mode": "per_document",
        "splits": {
            name: {
                "group_unit": doc_group_unit[name],
                "n_documents": int(len(doc_token_counts[name])),
                "tokens_total": int(sum(doc_token_counts[name])),
                "doc_token_counts": [int(c) for c in doc_token_counts[name]],
            }
            for name in doc_token_counts
        },
    }
    doc_boundaries_tmp = doc_boundaries_path + ".tmp"
    with open(doc_boundaries_tmp, "w", encoding="utf-8") as f:
        json.dump(doc_boundaries, f, ensure_ascii=False)
    doc_boundaries_sha256 = _sha256_file(doc_boundaries_tmp)
    data_report["structure"] = {
        "protocol": DOC_STRUCTURE_PROTOCOL,
        "encoding_mode": "per_document",
        "doc_boundaries_file": DOC_BOUNDARIES_FILE,
        "doc_boundaries_sha256": doc_boundaries_sha256,
        "ctx_grid": [int(c) for c in REPORT_CTX_GRID],
        "splits": structure_splits,
        "group_disjointness": group_audit,
        "note": (
            "token 流按文档群组分段编码：token 不跨文档边界，文档边界在 token "
            "空间精确可知。cross_document_windows 用训练采样器口径给出跨文档"
            "窗口的期望比例；doc_aligned_windows 给出完全落在单个文档内的"
            "窗口数与 token 覆盖量。两者共同界定“数据拼接结构”对结论的影响。"
        ),
    }
    # 短 n-gram 重叠作为诊断指标随报告落盘（告警在下面产生，不进入训练前
    # 硬闸门）：用于抓 64-gram 漏掉的短模板化复用。
    data_report["contamination_short_ngram"] = contamination_short
    data_report["contamination_exact"] = contamination_exact
    from .datacheck import short_overlap_alerts

    _salerts = short_overlap_alerts(contamination_short)
    if _salerts:
        data_report["alerts"] = list(data_report.get("alerts") or []) + _salerts
    _xalerts = exact_overlap_alerts(contamination_exact)
    if _xalerts:
        data_report["alerts"] = list(data_report.get("alerts") or []) + _xalerts
    # 访问账本只统计“评测级”test 读取；但 prepare_bins 为了编码 test.bin 与
    # 计算跨 split 重叠/分布漂移，本身会在建模前读一次 test（模型无关的诊断
    # 读取，不用于任何选择）。显式记录，避免账本 0 次被误读成“test 从未被碰”。
    data_report["test_reads_at_prep"] = {
        "encoded": bool(te is not None),
        "purpose": (
            "编码 test.bin + 跨 split 精确/采样 n-gram 重叠、"
            "全量精确 n-gram 扫描、近似重复与分布漂移诊断（模型无关）"
        ),
        "used_for_selection": False,
    }
    data_report["test_access"] = summarize_test_access(read_test_access_log(data_dir))

    # 原子替换：重建中断时不会让旧 meta 搭配半成品 bin 继续被当作有效缓存。
    train_tmp, val_tmp, meta_tmp = (
        train_bin + ".tmp",
        val_bin + ".tmp",
        meta_path + ".tmp",
    )
    tr.tofile(train_tmp)
    va.tofile(val_tmp)
    # bin 内容哈希：缓存命中时据此确认二进制流本身未被同尺寸替换/篡改，
    # 而不是只看 token 数量。
    bin_sha256 = {"train": _sha256_file(train_tmp), "val": _sha256_file(val_tmp)}
    if te is not None:
        test_tmp = test_bin + ".tmp"
        te.tofile(test_tmp)
        bin_sha256["test"] = _sha256_file(test_tmp)
    meta = {
        "tokenizer": tok.name,
        "requested_kind": kind,
        "vocab_size": tok.n_vocab,
        "fit_on": tokenizer_report["fit_on"],
        "text_sanitized": True,
        "cache_schema": int(CACHE_SCHEMA),
        "source_sha256": source_sha256,
        "split_set_digest": split_set_digest_value,
        "bin_sha256": bin_sha256,
        "near_dup": {
            "threshold": float(near_dup_threshold),
            "min_chars": int(near_dup_min_chars),
        },
        "train_tokens": int(tr.size),
        "val_tokens": int(va.size),
        # 文档结构证据的协议与哈希：文档边界侧车是 token 流的必要伴随物，
        # 缺了它就无法回答“评测窗口有多少跨过了文档边界”。
        "doc_structure_protocol": DOC_STRUCTURE_PROTOCOL,
        "doc_boundaries_sha256": doc_boundaries_sha256,
        "encoding_mode": "per_document",
        "group_disjoint": bool(group_audit.get("groups_disjoint")),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    if te is not None:
        meta["test_tokens"] = int(te.size)
    prov_digest = _source_provenance_digest(extra_report)
    if prov_digest is not None:
        # 来源身份进入缓存身份：原始 URL/revision/字节变化时，即使清洗后
        # 文本碰巧相同，报告里的溯源也必须同步更新，不能静默复用旧报告。
        meta["source_provenance_digest"] = prov_digest
    if tok.name == "char":
        meta["itos"] = tok.itos
        meta["char_unk"] = UNK_TOKEN in tok.stoi
    # 报告先落临时文件并哈希进 meta：缓存命中时若报告被改写/过期可被发现
    report_tmp = report_path + ".tmp"
    with open(report_tmp, "w", encoding="utf-8") as f:
        json.dump(data_report, f, ensure_ascii=False, indent=2)
    meta["report_sha256"] = _sha256_file(report_tmp)
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    # 替换顺序：bin/侧车/报告先落位，meta 最后（meta 是缓存有效性的判定依据；
    # 中断时留下的旧 meta 与新旧混合产物会被大小/哈希核对触发重建）。
    os.replace(train_tmp, train_bin)
    os.replace(val_tmp, val_bin)
    if te is not None:
        os.replace(test_tmp, test_bin)
    elif os.path.exists(test_bin):
        # 本次没有 test.txt 但残留旧 test.bin：删除，避免用过期 split 评测
        os.remove(test_bin)
    os.replace(doc_boundaries_tmp, doc_boundaries_path)
    os.replace(report_tmp, report_path)
    os.replace(meta_tmp, meta_path)
    msg = f"[data] train {tr.size:,} tokens / val {va.size:,} tokens"
    if te is not None:
        msg += f" / test {te.size:,} tokens"
    print(f"{msg} -> {data_dir}")
    for alert in data_report.get("alerts", []):
        tag = "警告" if alert.get("level") == "warning" else "提示"
        print(f"[data] {tag}：{alert.get('message')}")
    print(f"[data] 防泄漏报告 -> {report_path}")
    return meta


def load_meta(data_dir: str) -> dict:
    """读取 data_dir/meta.json；损坏或顶层非对象时给出明确错误。

    下游（分词器/训练/评测）一律按字段读取元数据，合法 JSON 但顶层是
    数组/标量（如 ``null``）时在此处显式拒绝，避免后续 ``meta["tokenizer"]``
    抛出难以定位的 TypeError。
    """
    path = os.path.join(data_dir, "meta.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except ValueError as e:
        raise ValueError(f"{path} 不是合法 JSON: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError(f"{path} 顶层不是 JSON 对象（实际为 {type(meta).__name__}）")
    return meta


def load_tokenizer(data_dir: str):
    meta = load_meta(data_dir)
    kind = meta.get("tokenizer")
    if not isinstance(kind, str) or not kind:
        raise ValueError(
            f"{os.path.join(data_dir, 'meta.json')} 缺少合法的 tokenizer 字段"
        )
    if kind == "gpt2_bpe":
        return get_gpt2_tokenizer()
    if kind != "char":
        raise ValueError(
            f"{os.path.join(data_dir, 'meta.json')} 的 tokenizer={kind!r} 未知："
            "只支持 'gpt2_bpe' 或 'char'（未知类型不能静默按字符级重建）"
        )
    itos = meta.get("itos")
    if (
        not isinstance(itos, list)
        or not itos
        or not all(isinstance(ch, str) for ch in itos)
    ):
        raise ValueError(
            f"{os.path.join(data_dir, 'meta.json')} 的 itos 不是非空字符串数组"
        )
    # 新缓存显式携带 <unk>（char_unk=True）；旧缓存没有该标记，按原编码方式
    # 重建（未知字符→0），保证与旧 bin 的 id 一一对应。
    return CharTokenizer(
        itos, add_unk=config_bool(meta.get("char_unk", False), "char_unk")
    )


def load_doc_boundaries(data_dir: str):
    """读取 doc_boundaries.json（文档边界侧车）；缺失/损坏返回 None。

    侧车记录每个 split 的逐文档 token 计数：它是“窗口跨了多少次文档边界”
    与“文档内 PPL”这一独立口径的唯一依据。缺失时下游必须如实标记为
    “结构证据不可用”，不能默认“没有跨文档窗口”。
    """
    path = os.path.join(data_dir, DOC_BOUNDARIES_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def doc_token_counts_for_split(data_dir: str, split: str):
    """取某个 split 的逐文档 token 计数。

    返回 ``(counts, meta)``：``counts`` 为 None 表示结构证据不可用（侧车
    缺失、损坏或不含该 split）；``meta`` 为该 split 的侧车元信息。
    """
    if not isinstance(split, str) or split not in ("train", "val", "test"):
        raise ValueError(f"split 必须是 'train'/'val'/'test' 之一，收到 {split!r}")
    payload = load_doc_boundaries(data_dir)
    if not payload:
        return None, None
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        return None, None
    entry = splits.get(split)
    if not isinstance(entry, dict):
        return None, None
    raw = entry.get("doc_token_counts")
    if not isinstance(raw, list):
        return None, None
    try:
        counts = [int(c) for c in raw]
    except (TypeError, ValueError):
        return None, None
    if any(c < 0 for c in counts):
        return None, None
    return counts, entry


def load_bin(data_dir: str, split: str, allow_test: bool = False) -> np.memmap:
    """读取 token 流；test split 默认拒绝（尘封）。

    测试集只允许在最终一次性评估中通过 lega.evaluate 访问（它同时写入
    .test_access_log.jsonl 访问账本）。训练、调参、选型代码调用
    load_bin(..., "test") 时必须显式传 allow_test=True，把“读测试集”
    从默认行为变成需要主动声明的动作。
    """
    # split 必须是封闭枚举，而不是任意路径片段：`"./test"`、`"test/../test"`
    # 或绝对路径都会绕过上面的 `split == "test"` 尘封闸门（os.path.join 对
    # 绝对路径还会丢弃 data_dir）。只允许数据协议定义的三个 split。
    if not isinstance(split, str) or split not in ("train", "val", "test"):
        raise ValueError(f"split 必须是 'train'/'val'/'test' 之一，收到 {split!r}")
    if split == "test" and allow_test is not True:
        # 只接受真正的 bool True：字符串 "false"、1 等真值若按 truthiness
        # 放行，会把“显式开锁”的闸门变成任何非空值都能绕过。
        raise PermissionError(
            "测试集已尘封：load_bin(split='test') 需显式 allow_test=True"
            "（不接受非布尔真值）；test 只用于最终一次性评估"
            "（lega.evaluate 会写访问账本），训练/调参/选型不得读取"
        )
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(
            f"{path} 不存在或为空，请先运行 prepare_bins / prepare_data.py"
        )
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = load_meta(data_dir)
        recorded = meta.get("bin_sha256")
        # 纯 bin 目录（旧格式 meta 无内容哈希）按不可判定放行；记录过哈希
        # 就必须一致，否则 bin 已被同尺寸替换/篡改，诊断报告不再对应。
        if isinstance(recorded, dict) and split in recorded:
            if _sha256_file(path) != recorded[split]:
                raise ValueError(
                    f"{path} 内容与 meta.json 记录的 bin 哈希不一致："
                    "bin 被替换/篡改/损坏，请重新运行 prepare_bins 重建"
                )
    return np.memmap(path, dtype=np.uint16, mode="r")


# ---------------------------------------------------------------- 批采样

# H2D 乒乓暂存：键 (device, shape) -> [(pinned_buf, event), (pinned_buf, event)]。
# 双槽轮换 + event 保证重写槽位前上一次异步拷贝已完成。
_H2D_STAGE: dict = {}
_H2D_TURN: dict = {}
_H2D_STAGE_BYTES = 0
_H2D_STAGE_MAX_BYTES = 128 * 1024 * 1024


def _stage_to_device(src, device: str) -> torch.Tensor:
    """把 (n, ctx+1) 的 token id 块搬运到目标设备并升为 int64。

    src 可以是已物化的 np.ndarray，也可以是 (memmap, span) 二元组：
    CUDA 目标把 memmap 的花式索引进同 dtype 的 pinned 输出缓冲（np.take 的
    out= 参数），再异步拷贝并在设备端升为 int64；CPU 目标先按源 dtype 做
    花式索引，再写入 int64 缓冲（不能跨 dtype 使用 np.take 的 out=）。
    """
    global _H2D_STAGE_BYTES
    is_span = isinstance(src, tuple)
    shape = src[1].shape if is_span else src.shape
    dtype = np.uint16 if is_span else src.dtype
    if not device.startswith("cuda"):
        if is_span:
            # 花式索引先按源 dtype（uint16）取值，再写入 int64 缓冲。不能把
            # int64 缓冲直接交给 np.take(out=...)：numpy 要求 out 与源满足
            # safe-cast，跨 dtype 会抛 TypeError（旧实现依赖未定义行为）。
            arr = np.empty(shape, dtype=np.int64)
            arr[...] = np.asarray(src[0])[src[1]]
            return torch.from_numpy(arr)
        arr = src if src.flags["C_CONTIGUOUS"] else np.ascontiguousarray(src)
        out = torch.from_numpy(arr)
        return out if out.dtype == torch.int64 else out.long()
    key = (device, tuple(shape), dtype)
    slots = _H2D_STAGE.get(key)
    if slots is None:
        tdtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        slot_bytes = torch.empty(0, dtype=tdtype).element_size()
        slot_bytes *= 2 * int(np.prod(shape, dtype=np.int64))
        if (
            len(_H2D_STAGE) < 32
            and _H2D_STAGE_BYTES + slot_bytes <= _H2D_STAGE_MAX_BYTES
        ):
            slots = [
                (torch.empty(shape, dtype=tdtype).pin_memory(), torch.cuda.Event())
                for _ in range(2)
            ]
            _H2D_STAGE[key] = slots
            _H2D_TURN[key] = 0
            _H2D_STAGE_BYTES += slot_bytes
        else:
            arr = np.take(src[0], src[1]) if is_span else src
            out = torch.from_numpy(np.ascontiguousarray(arr)).to(
                device, non_blocking=True
            )
            return out if out.dtype == torch.int64 else out.long()
    turn = _H2D_TURN[key]
    buf, ev = slots[turn]
    _H2D_TURN[key] = 1 - turn
    ev.synchronize()  # 等待该槽上一轮的 H2D 拷贝读完 buf
    if is_span:
        # memmap 花式索引进 pinned 缓冲的 np 视图
        np.take(src[0], src[1], out=buf.cpu().numpy())
    else:
        buf.copy_(torch.from_numpy(src))
    out = buf.to(device, non_blocking=True)
    ev.record()
    return out if out.dtype == torch.int64 else out.long()


# 按 ctx 缓存索引基底
_SPAN_BASE: dict = {}


def _span_base(ctx: int) -> np.ndarray:
    base = _SPAN_BASE.get(ctx)
    if base is None:
        base = np.arange(ctx + 1)
        _SPAN_BASE[ctx] = base
    return base


# 宿主端跨度暂存：按 (n_rows, ctx) 缓存一份 int64 缓冲，跨度数组经 np.add 就地写入。
_SPAN_BUF: dict = {}


def _make_span(rows: np.ndarray, ctx: int) -> np.ndarray:
    """row start 一维数组 + 基底 -> (n, ctx+1) 跨度；结果写入按形状缓存的暂存。

    调用方必须在下一次同形状调用之前消费完（np.take / 花式索引均为同步读取），
    现有 _stage_to_device 的使用方式满足该约定。
    """
    key = (rows.shape[0], ctx)
    buf = _SPAN_BUF.get(key)
    if buf is None:
        buf = np.empty((rows.shape[0], ctx + 1), dtype=np.int64)
        _SPAN_BUF[key] = buf
    np.add(rows[:, None], _span_base(ctx), out=buf)
    return buf


def get_train_batch(
    train: np.memmap,
    ctx: int,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
    order_hasher=None,
):
    """随机偏移采样 (x, y)，y 为 x 右移一位。

    order_hasher（可选）接收本批窗口起点序列的规范化字节流。训练脚本用它
    记录**实际采样顺序**的 SHA-256：仅比较 seed/config 不足以证明各变体
    真的消费了同一批训练数据（例如采样代码路径被改、RNG 被提前消耗），
    而数据顺序不同会让“方法对比”混入数据混杂。

    ctx/batch_size 是采样形状身份：bool/小数经 int() 截断或负值会让
    ``len(train) - ctx`` 变成另一个采样范围/形状，必须在构造索引前拒绝。
    """
    ctx = strict_int(ctx, "ctx")
    batch_size = strict_int(batch_size, "batch_size")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx}")
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
    if len(train) < ctx + 1:
        raise ValueError(f"训练集只有 {len(train)} tokens，不足以采样 ctx={ctx}")
    ix = rng.integers(0, len(train) - ctx, size=batch_size)
    if order_hasher is not None:
        # 固定小端 int64，保证跨平台/跨进程的指纹口径一致。
        order_hasher.update(np.asarray(ix, dtype="<i8").tobytes())
    # 取 (batch_size, ctx+1) 的索引；uint16 贯穿到 H2D，设备端再转 long；
    # (train, span) 元组让 memmap 花式索引写入 pinned 暂存缓冲
    span = _make_span(ix, ctx)
    both = _stage_to_device((train, span), device)
    return both[:, :-1], both[:, 1:]


def get_train_batch_doc_aligned(
    train: np.memmap,
    ctx: int,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
    plan: dict,
    order_hasher=None,
):
    """按文档权重抽取“窗口不跨文档”的训练批次。

    与 :func:`get_train_batch` 的唯一差别是窗口（输入 + 标签）必须完整落在
    同一篇文档内；文档混合比例保持不变（文档权重 ∝ 文档 token 数），因此
    “窗口是否跨文档”成为可单独操控的受控因子，而不是与文档长度分布纠缠在
    一起的隐式差异。

    plan 由 :func:`lega.datacheck.doc_aligned_sampling_plan` 构造；缺少文档
    边界证据时无法保证“不跨文档”，必须显式失败而不是退回拼接流采样。
    """
    ctx = strict_int(ctx, "ctx")
    batch_size = strict_int(batch_size, "batch_size")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx}")
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
    if not isinstance(plan, dict):
        raise ValueError("plan 必须是由 doc_aligned_sampling_plan 构造的采样计划")
    doc_starts = plan.get("doc_starts")
    doc_tokens = plan.get("doc_tokens")
    doc_valid = plan.get("doc_valid_starts")
    cum_tokens = plan.get("cum_tokens")
    if doc_starts is None or doc_tokens is None or doc_valid is None:
        raise ValueError("plan 缺少文档边界数组（doc_starts/doc_tokens/...）")
    n_docs = int(np.asarray(doc_starts).size)
    if n_docs == 0:
        raise ValueError(
            f"没有任何文档能承载 ctx={ctx} 的窗口（需要至少 {ctx + 1} 个 token）："
            "无法在“窗口不跨文档”的口径下训练；请改用 train_sampler=stream "
            "或换用更小的 train_ctx"
        )
    if cum_tokens is None:
        cum_tokens = np.cumsum(np.asarray(doc_tokens, dtype=np.int64))
    cum = np.asarray(cum_tokens, dtype=np.int64)
    total = int(cum[-1])
    if total <= 0:
        raise ValueError("采样计划的 token 总量为 0：无法采样")
    # 文档选择：权重 ∝ 文档 token 数（与拼接流均匀采样同一份文档混合比例）。
    u = rng.random(batch_size) * total
    idx = np.searchsorted(cum, u, side="right")
    np.clip(idx, 0, n_docs - 1, out=idx)
    # 文档内偏移：在该文档的合法起点区间 [0, n_tokens - ctx) 上均匀取值。
    valid = np.asarray(doc_valid, dtype=np.int64)[idx]
    offsets = rng.integers(0, valid)
    starts = np.asarray(doc_starts, dtype=np.int64)[idx] + offsets
    if order_hasher is not None:
        order_hasher.update(np.asarray(starts, dtype="<i8").tobytes())
    span = _make_span(starts, ctx)
    both = _stage_to_device((train, span), device)
    return both[:, :-1], both[:, 1:]


def make_train_batch_sampler(
    kind: str,
    train: np.memmap,
    ctx: int,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
    plan: dict = None,
):
    """返回训练批次采样函数（闭包），统一 ``stream`` / ``doc_aligned`` 口径。

    kind 是**受控因子**：``stream``（默认，起点在拼接流上均匀随机，窗口可能
    跨文档）与 ``doc_aligned``（窗口严格落在单个文档内）。两者返回同样的
    ``(x, y)``，调用方无需分支；order_hasher 语义也一致（记录实际消费的窗口
    起点序列），因此“同 seed 下各变体消费了同一批训练窗口”的审计在两种口径
    下都成立。
    """
    if not isinstance(kind, str) or kind not in ("stream", "doc_aligned"):
        raise ValueError(
            f"train_sampler 必须是 'stream'/'doc_aligned' 之一，收到 {kind!r}"
        )
    if kind == "stream":

        def sampler(order_hasher=None):
            return get_train_batch(
                train, ctx, batch_size, device, rng, order_hasher=order_hasher
            )

        return sampler
    if plan is None:
        raise ValueError(
            "train_sampler=doc_aligned 需要文档边界采样计划（plan），"
            "缺少 doc_boundaries.json 时不能保证窗口不跨文档"
        )

    def sampler(order_hasher=None):
        return get_train_batch_doc_aligned(
            train, ctx, batch_size, device, rng, plan, order_hasher=order_hasher
        )

    return sampler


def get_val_batch(val: np.memmap, ctx: int, idx: int, device: str):
    """顺序不重叠窗口，idx 为窗口序号（循环取用）；等价于窗口数为 1 的批量版。"""
    return get_val_windows(val, ctx, (idx,), device)


def get_val_windows(val: np.memmap, ctx: int, indices, device: str):
    """一次取多个不重叠窗口拼成批量 (n, ctx)。

    窗口序号是 ``s·ctx`` 的非负下标（与 ``_val_eval_indices`` 同口径）；
    负数/小数/bool 不是合法窗口身份，不能靠取模静默映射到另一个窗口。
    标量序号按单窗口处理（``get_val_batch`` 的元组调用保持原语义）。
    """
    ctx = strict_int(ctx, "ctx")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx}")
    try:
        raw = np.atleast_1d(np.asarray(indices))
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"indices 必须是整数窗口序号，收到 {indices!r}") from e
    int64_max = int(np.iinfo(np.int64).max)
    if raw.dtype.kind in "iu":
        if raw.size and (int(raw.min()) < 0 or int(raw.max()) > int64_max):
            raise ValueError(
                f"窗口序号必须是 [0, {int64_max}] 内的整数，收到 {indices!r}"
            )
        idx = raw.astype(np.int64, copy=False)
    else:
        # 浮点/对象/字符串逐元素走 strict_int：1.5/True/"1_0" 不能被
        # np.asarray(..., dtype=int64) 静默截断/当成 1/10。
        vals = []
        for v in raw.reshape(-1).tolist():
            try:
                iv = strict_int(v, "窗口序号")
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(f"窗口序号必须是整数，收到 {indices!r}") from e
            if iv < 0 or iv > int64_max:
                raise ValueError(
                    f"窗口序号必须在 [0, {int64_max}] 内，收到 {indices!r}"
                )
            vals.append(iv)
        idx = np.asarray(vals, dtype=np.int64)
    n_windows = (len(val) - 1) // max(ctx, 1)
    if n_windows <= 0:
        raise ValueError(f"验证集只有 {len(val)} tokens，不足以组成 ctx={ctx} 的窗口")
    # 允许 idx >= n_windows 的循环取用；负值已在上方拒绝。
    starts = (idx % n_windows) * ctx
    span = _make_span(starts, ctx)
    # (val, span) 元组：memmap 花式索引直写 pinned 缓冲，设备端升 long 并切 x/y
    both = _stage_to_device((val, span), device)
    return both[:, :-1], both[:, 1:]


def get_val_windows_at(val: np.memmap, ctx: int, starts, device: str):
    """按显式 token 起点取验证窗口（用于非 ctx 整数倍的相位平移诊断）。

    与 ``get_val_windows`` 不同，起点不做取模/对齐：相位平移后的窗口起点可以
    不是 ctx 的整数倍；越界/负数/bool/小数必须显式拒绝，否则会静默读取另一
    段验证文本。返回 ``(x, y)``，y 为 x 右移一位。
    """
    ctx = strict_int(ctx, "ctx")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx}")
    try:
        raw = np.atleast_1d(np.asarray(starts))
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"starts 必须是整数 token 下标，收到 {starts!r}") from e
    int64_max = int(np.iinfo(np.int64).max)
    if raw.dtype.kind in "iu":
        if raw.size and (int(raw.min()) < 0 or int(raw.max()) > int64_max):
            raise ValueError(
                f"窗口起点必须是 [0, {int64_max}] 内的整数，收到 {starts!r}"
            )
        pos = raw.astype(np.int64, copy=False)
    else:
        vals = []
        for v in raw.reshape(-1).tolist():
            try:
                iv = strict_int(v, "窗口起点")
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(f"窗口起点必须是整数，收到 {starts!r}") from e
            if iv < 0 or iv > int64_max:
                raise ValueError(f"窗口起点必须在 [0, {int64_max}] 内，收到 {starts!r}")
            vals.append(iv)
        pos = np.asarray(vals, dtype=np.int64)
    max_start = len(val) - ctx - 1
    if max_start < 0:
        raise ValueError(f"验证集只有 {len(val)} tokens，不足以组成 ctx={ctx} 的窗口")
    if pos.size and int(pos.max()) > max_start:
        raise ValueError(
            f"窗口起点最大 {max_start}（len(val)={len(val)}, ctx={ctx}），"
            f"收到 {starts!r}"
        )
    span = _make_span(pos, ctx)
    both = _stage_to_device((val, span), device)
    return both[:, :-1], both[:, 1:]
