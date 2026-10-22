"""数据管线：分词器（GPT-2 BPE，tiktoken 不可用时退化为字符级）+ bin 缓存 + 批采样。

防泄露约定（细节见 lega/datacheck.py）:
  * 先划分 train/val/test，再做编码等预处理；字符级词表**只在 train 上拟合**，
    val/test 只做 transform，未见字符统一映射 <unk>（BPE 为公开预训练词表，
    不依赖本项目数据，不存在此问题）；
  * 编码前后做硬校验（整数、有限、id 越界）与防泄漏诊断（重复行去污染、
    64-gram 重叠筛查、一元分布漂移），结果写 data_report.json；
  * val/test 内容绝不被改写；测试集只在最终评估时访问（访问账本）。

约定 data_dir 内文件：
  train.txt / val.txt     原始文本（prepare_data.py 或 make_dummy_data.py 生成）
  train.bin / val.bin     uint16 token 流（prepare_bins 生成）
  meta.json               {"tokenizer": "gpt2_bpe"|"char", "vocab_size": int, ...}
  data_report.json        防泄漏/清洗/漂移诊断报告
"""
import hashlib
import json
import math
import os
from typing import List

import numpy as np
import torch

from .datacheck import (UNK_TOKEN, build_data_report, check_vocab_capacity,
                        decontaminate_documents, decontaminate_texts,
                        distribution_shift, ngram_overlap,
                        read_test_access_log, sanitize_text,
                        short_ngram_overlap,
                        summarize_test_access, text_length_stats,
                        token_stream_stats, unknown_char_rate)
from .utils import config_bool, strict_int

_GPT2_CACHE = {}

# bin 缓存协议版本：清洗/去污染/词表拟合逻辑变更时必须 +1，
# 旧缓存（含未记录版本的第一代缓存）会被强制重建，
# 否则会用旧协议产出的 bin 静默训练，防泄漏自检报告也与实际数据不符。
# v6：meta 记录 train/val/test.bin 的内容哈希，缓存命中与 load_bin 均核对，
# 防止同尺寸替换 bin 后绕过源文本/报告哈希核对。
CACHE_SCHEMA = 6


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


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


def _cache_integrity(data_dir: str, meta: dict, report_path: str,
                     report_enabled: bool):
    """缓存完整性核对：源文本哈希 + 报告哈希 + bin 内容哈希 + 缓存协议版本。

    返回 (ok, issue, has_sources)。缓存命中只核对大小无法发现“txt 改了但
    bin 没重建”：那样训练集张冠李戴，data_report.json 里的去污染/重叠/
    漂移诊断也不再对应当前数据。只要存在源文本就强校验；只分发 bin（无
    任何 txt）的场景无法核对也无法重建，按不可判定放行，由调用方提示。
    """
    has_sources = any(os.path.exists(os.path.join(data_dir, f"{s}.txt"))
                      for s in ("train", "val", "test"))
    bin_ok = _bin_hashes_ok(data_dir, meta)
    if bin_ok is False:
        return (False, "bin 内容与构建时记录的哈希不一致"
                       "（被替换/篡改/损坏），重建以恢复与源文本一致的 token 流",
                False)
    if not has_sources:
        # 不可判定放行，并把 no_sources 如实返回给调用方（用于提示与缓存判定）
        return True, None, True
    try:
        cache_schema = strict_int(meta.get("cache_schema", 0),
                                  "meta.cache_schema")
    except (TypeError, ValueError, OverflowError):
        cache_schema = -1
    if cache_schema != CACHE_SCHEMA:
        return (False, "缓存由旧版数据管线生成（无版本/源哈希记录），"
                       "重建以应用最新清洗与去污染协议", False)
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
    """tiktoken gpt2 的轻量包装，统一 encode/decode/n_vocab 接口。"""

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

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def get_gpt2_tokenizer() -> GPT2Tokenizer:
    if "tok" not in _GPT2_CACHE:
        import tiktoken
        _GPT2_CACHE["tok"] = GPT2Tokenizer(tiktoken.get_encoding("gpt2"))
    return _GPT2_CACHE["tok"]


def get_tokenizer(kind: str, train_text: str = "", val_text: str = ""):
    """kind ∈ {gpt2, auto}；gpt2 失败时 auto 退化为字符级，gpt2 直接抛错。

    字符级词表只在 train_text 上拟合（val_text 仅为兼容旧调用保留、不参与），
    这保证 val/test 不通过词表构造向训练侧泄露统计信息。
    """
    if kind in ("gpt2", "auto"):
        try:
            return get_gpt2_tokenizer()
        except Exception as e:  # 无网络且无本地缓存
            if kind == "gpt2":
                raise
            print(f"[data] tiktoken 不可用({e})，退化为字符级分词")
    chars = sorted(set(train_text))
    if not chars:
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,\n")
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
    return ((str(tokenizer) == "char" and fit_on == "train")
            or (str(tokenizer) == "gpt2_bpe" and fit_on == "gpt2_pretrained"))


def prepare_bins(data_dir: str, kind: str = "auto", force: bool = False,
                 report: bool = True, extra_report: dict = None) -> dict:
    """把 train.txt/val.txt（以及若存在的 test.txt）编码为 uint16 bin
    （已存在则跳过，force=True 重建）。

    这是所有文本进入训练/评测的唯一入口，因此在此集中执行防泄漏处理：
      1) 文本清洗（坏字符）与跨 split 重复行去污染（只改 train）；
      2) 分词器只在 train 上拟合（字符级词表 + <unk>）；
      3) val/test 只 transform，随后对三条 token 流做硬校验；
      4) 生成 data_report.json（重复、64-gram 重叠、分布漂移、访问账本）。
    extra_report 用于记录数据来源/划分协议等上游元数据（原样并入报告）。
    """
    meta_path = os.path.join(data_dir, "meta.json")
    report_path = os.path.join(data_dir, "data_report.json")
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    test_txt_path = os.path.join(data_dir, "test.txt")
    has_test_txt = os.path.exists(test_txt_path)
    test_bin = os.path.join(data_dir, "test.bin")
    if (os.path.exists(meta_path) and os.path.exists(train_bin)
            and os.path.exists(val_bin) and not force):
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
        # 旧缓存可能是在 val/test 上拟合词表（泄露）或缺少防泄漏报告，必须重建
        fit_ok = tokenizer_fit_source_ok(meta.get("tokenizer"),
                                         meta.get("fit_on"))
        integrity_ok, integrity_issue, no_sources = _cache_integrity(
            data_dir, meta, report_path, report)

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

        train_valid = (os.path.getsize(train_bin)
                       == _meta_int("train_tokens") * 2)
        val_valid = (os.path.getsize(val_bin)
                     == _meta_int("val_tokens") * 2)
        # 无源 txt（纯 bin 数据目录）时无法判定 test.bin 是否对应缺失的
        # test.txt：显式保留 test.bin 视为有效，否则该场景会在重建路径上
        # 因缺少 train.txt 直接报错。
        test_valid = (not has_test_txt and (
            no_sources or not os.path.exists(test_bin))) or (
            has_test_txt
            and os.path.exists(test_bin)
            and os.path.getsize(test_bin) == _meta_int("test_tokens") * 2)
        if (cached_kind == kind or kind == "auto") and train_valid and val_valid \
                and test_valid and fit_ok and integrity_ok:
            if no_sources:
                print("[data] 警告：数据目录只有 bin 没有源 txt，"
                      "无法核对缓存与源文本一致性（按不可判定放行）")
            if kind == "auto":
                # auto 的实际分词器取决于当前环境（tiktoken 是否可用），
                # 直接复用缓存可能与本应使用的分词器不一致，必须提示。
                try:
                    now_tok = get_gpt2_tokenizer().name
                except Exception:
                    now_tok = "char"
                if meta.get("tokenizer") != now_tok:
                    print(f"[data] 警告：kind=auto 复用了 {meta.get('tokenizer')} 缓存，"
                          f"但当前环境 auto 将使用 {now_tok}；如需一致请 force=True 重建")
            return meta
        if not (train_valid and val_valid and test_valid):
            print("[data] 缓存 bin/meta 大小不一致，重建 bin")
        elif not fit_ok:
            print("[data] 旧缓存词表可能在全量数据上拟合（泄露风险），重建 bin")
        elif not integrity_ok:
            print(f"[data] {integrity_issue}，重建 bin")
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

    # 1) 清洗 + 去污染：坏字符确定性删除/替换；与 val/test 精确重复的长行
    #    只从 train 移除（val/test 绝不改写，保持尘封）。
    sanitization = {}
    for name in list(texts):
        clean, stats = sanitize_text(texts[name])
        texts[name] = clean
        sanitization[name] = stats
    # 块级去污染先于行级：整块复制（可能由多行短行组成）一旦被行级先删掉
    # 其中一条长行，块哈希就不再与 eval 匹配，剩余短行会漏检。
    texts, document_stats = decontaminate_documents(texts)
    texts, duplicate_stats = decontaminate_texts(texts, min_chars=40)
    # 行级删除会改变块结构：删掉块中的长行后，剩余短行可能恰好重新拼成
    # 与 val/test 完全相同的多行块。只做“先块后行”会漏掉这种新生成的
    # 副本，因此行级确有删除时再跑一遍块级去污染（只会继续删，必然终止）。
    line_dropped = sum(duplicate_stats[k] for k in (
        "train_lines_dropped_as_duplicates_of_eval",
        "train_lines_dropped_as_normalized_duplicates_of_eval",
        "train_intra_split_duplicate_lines_removed",
        "train_intra_split_normalized_duplicate_lines_removed"))
    if line_dropped:
        texts, post_doc_stats = decontaminate_documents(texts)
        for key in ("train_blocks_dropped_as_duplicates_of_eval",
                    "train_intra_split_duplicate_blocks_removed",
                    "train_lines_dropped", "train_chars_dropped"):
            document_stats[key] = (int(document_stats.get(key, 0))
                                   + int(post_doc_stats.get(key, 0)))
    duplicate_stats["document_level"] = document_stats
    train_text, val_text, test_text = texts["train"], texts["val"], texts.get("test")
    # 原文此后只由局部变量引用：逐段编码完即释放
    del texts
    n_doc_cross = document_stats[
        "train_blocks_dropped_as_duplicates_of_eval"]
    n_doc_intra = document_stats[
        "train_intra_split_duplicate_blocks_removed"]
    if n_doc_cross or n_doc_intra:
        print(f"[data] 块级去污染：从 train 移除 {n_doc_cross:,} 个与 val/test "
              f"重复的多行块、{n_doc_intra:,} 个内部重复块"
              f"（共 {document_stats['train_lines_dropped']:,} 行；"
              "val/test 不改写）")
    if document_stats["val_test_overlap_blocks"]:
        print(f"[data] 警告：val 与 test 有 "
              f"{document_stats['val_test_overlap_blocks']:,} 个多行块重复"
              "（两侧均未改写；val 选型对 test 的独立性下降）")
    n_cross = (duplicate_stats["train_lines_dropped_as_duplicates_of_eval"]
               + duplicate_stats[
                   "train_lines_dropped_as_normalized_duplicates_of_eval"])
    n_intra = (duplicate_stats[
                   "train_intra_split_duplicate_lines_removed"]
               + duplicate_stats[
                   "train_intra_split_normalized_duplicate_lines_removed"])
    if n_cross:
        print(f"[data] 去污染：从 train 移除 {n_cross:,} 条与 val/test "
              "重复（精确/空白归一化）的长行")
    if n_intra:
        print(f"[data] 去重：从 train 移除 {n_intra:,} 条内部重复长行"
              "（保留首次出现；val/test 不改写）")
    if duplicate_stats["val_test_overlap_long_lines"]:
        print(f"[data] 警告：val 与 test 有 "
              f"{duplicate_stats['val_test_overlap_long_lines']:,} 条长行重复"
              "（两侧均未改写；val 选型对 test 的独立性下降）")

    # 2) 分词器只在 train 上拟合；3) val/test 只做 transform
    tok = get_tokenizer(kind, train_text)
    check_vocab_capacity(tok.n_vocab)
    print(f"[data] 分词器: {tok.name}（词表仅在 train 上拟合）, vocab={tok.n_vocab}")

    unk_rates = {}
    for name, txt in (("train", train_text), ("val", val_text), ("test", test_text)):
        if txt is not None:
            unk_rates[name] = unknown_char_rate(tok, txt)
    if unk_rates.get("test") is not None and unk_rates["test"] > 0.001:
        print(f"[data] 提示：test 字符级 OOV 率 {unk_rates['test']:.4%}，"
              "未见字符映射 <unk>（词表只用 train 拟合，防止泄露）")

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
                        f"拒绝编码 {pos}:{end}")
                wide = arr.astype(np.int64)
                lo, hi = int(wide.min()), int(wide.max())
                if lo < 0 or hi > 65535:
                    # 先 astype(uint16) 再校验会让 70000 环绕成 4464，
                    # token_stream_stats 只看到合法 id，脏数据被静默接受。
                    raise ValueError(
                        f"token id 越界 uint16（min={lo}, max={hi}，"
                        f"分块 {pos}:{end}）：拒绝静默环绕，请检查分词器/词表")
                parts.append(wide.astype(np.uint16))
            else:
                parts.append(np.zeros(0, dtype=np.uint16))
            pos = end
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)

    tr = encode_stream(train_text)
    text_stats = {"train": text_length_stats(train_text)}
    train_text = None
    va = encode_stream(val_text)
    text_stats["val"] = text_length_stats(val_text)
    val_text = None
    te = None
    if test_text is not None:
        te = encode_stream(test_text)
        text_stats["test"] = text_length_stats(test_text)
        test_text = None

    # 4) token 流硬校验 + 防泄漏诊断
    streams = {"train": tr, "val": va}
    if te is not None:
        streams["test"] = te
    stream_stats = {name: token_stream_stats(arr, tok.n_vocab, name)
                    for name, arr in streams.items()}
    # 字符级分词器把所有未见字符映射到同一个 <unk>：不同 split 的 OOV 长串
    # 会在 token 层变成完全相同的 0 序列，若直接做 n-gram 重叠会被误判为
    # “语料复制”并触发 hard fail。排除含 unk 的窗口，只让真实 token 作为
    # 泄漏证据（原文级重复仍由行/块去污染与 verify_splits_disjoint 检查）。
    ignore_ids = None
    if tok.name == "char" and UNK_TOKEN in getattr(tok, "stoi", {}):
        ignore_ids = {int(tok.unk_id)}
    contamination = {"train_val": ngram_overlap(
        tr, va, ignore_ids=ignore_ids)}
    # 短 n-gram 筛查：64-gram 会漏掉“整句/整段复用但 <64 token”的模板化泄露；
    # 短 n 更易命中自然搭配，因此只作诊断告警（见 leakage_alerts）。
    contamination_short = {"train_val": short_ngram_overlap(
        tr, va, ignore_ids=ignore_ids)}
    distribution = {"val_vs_train": distribution_shift(tr, va, tok.n_vocab)}
    if te is not None:
        contamination["train_test"] = ngram_overlap(
            tr, te, ignore_ids=ignore_ids)
        # eval 两侧也要互查：val 用于选 checkpoint，test 用于最终报告；
        # 二者若有共享片段，val 选择会部分“预演”test，test 不再独立。
        contamination["val_test"] = ngram_overlap(
            va, te, ignore_ids=ignore_ids)
        contamination_short["train_test"] = short_ngram_overlap(
            tr, te, ignore_ids=ignore_ids)
        contamination_short["val_test"] = short_ngram_overlap(
            va, te, ignore_ids=ignore_ids)
        distribution["test_vs_train"] = distribution_shift(tr, te, tok.n_vocab)
        distribution["test_vs_val"] = distribution_shift(va, te, tok.n_vocab)
    tokenizer_report = {
        "kind": tok.name,
        "fit_on": "train" if tok.name == "char" else "gpt2_pretrained",
        "vocab_size": int(tok.n_vocab),
        "unk_token": (UNK_TOKEN if getattr(tok, "stoi", {}).get(UNK_TOKEN)
                      is not None else None),
        "unk_rate": {k: v for k, v in unk_rates.items() if v is not None},
    }
    data_report = build_data_report(
        tokenizer=tokenizer_report, splits=stream_stats,
        sanitization=sanitization, duplicates=duplicate_stats,
        contamination={k: v for k, v in contamination.items()},
        distribution={k: v for k, v in distribution.items()},
        text_stats=text_stats, extra=extra_report)
    # 短 n-gram 重叠作为诊断指标随报告落盘（告警在下面产生，不进入训练前
    # 硬闸门）：用于抓 64-gram 漏掉的短模板化复用。
    data_report["contamination_short_ngram"] = contamination_short
    from .datacheck import short_overlap_alerts
    _salerts = short_overlap_alerts(contamination_short)
    if _salerts:
        data_report["alerts"] = list(data_report.get("alerts") or []) + _salerts
    # 访问账本只统计“评测级”test 读取；但 prepare_bins 为了编码 test.bin 与
    # 计算跨 split 重叠/分布漂移，本身会在建模前读一次 test（模型无关的诊断
    # 读取，不用于任何选择）。显式记录，避免账本 0 次被误读成“test 从未被碰”。
    data_report["test_reads_at_prep"] = {
        "encoded": bool(te is not None),
        "purpose": "编码 test.bin + 跨 split 重叠/分布漂移诊断（模型无关）",
        "used_for_selection": False,
    }
    data_report["test_access"] = summarize_test_access(
        read_test_access_log(data_dir))

    # 原子替换：重建中断时不会让旧 meta 搭配半成品 bin 继续被当作有效缓存。
    train_tmp, val_tmp, meta_tmp = train_bin + ".tmp", val_bin + ".tmp", meta_path + ".tmp"
    tr.tofile(train_tmp)
    va.tofile(val_tmp)
    # bin 内容哈希：缓存命中时据此确认二进制流本身未被同尺寸替换/篡改，
    # 而不是只看 token 数量。
    bin_sha256 = {"train": _sha256_file(train_tmp), "val": _sha256_file(val_tmp)}
    if te is not None:
        test_tmp = test_bin + ".tmp"
        te.tofile(test_tmp)
        bin_sha256["test"] = _sha256_file(test_tmp)
    # 源文本指纹：缓存命中时据此确认 bin 确实由当前 train/val/test.txt 构建，
    # 防止“txt 改了但 bins 没重建”的静默过期（防泄漏报告也会随之失真）。
    source_sha256 = {}
    for split, path in (("train", os.path.join(data_dir, "train.txt")),
                        ("val", os.path.join(data_dir, "val.txt")),
                        ("test", test_txt_path)):
        if os.path.exists(path):
            source_sha256[split] = _sha256_file(path)
    meta = {"tokenizer": tok.name, "requested_kind": kind,
            "vocab_size": tok.n_vocab,
            "fit_on": tokenizer_report["fit_on"],
            "text_sanitized": True,
            "cache_schema": int(CACHE_SCHEMA),
            "source_sha256": source_sha256,
            "bin_sha256": bin_sha256,
            "train_tokens": int(tr.size), "val_tokens": int(va.size)}
    if te is not None:
        meta["test_tokens"] = int(te.size)
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
    # 替换顺序：bin/报告先落位，meta 最后（meta 是缓存有效性的判定依据；
    # 中断时留下的旧 meta 与新旧混合产物会被大小/哈希核对触发重建）。
    os.replace(train_tmp, train_bin)
    os.replace(val_tmp, val_bin)
    if te is not None:
        os.replace(test_tmp, test_bin)
    elif os.path.exists(test_bin):
        # 本次没有 test.txt 但残留旧 test.bin：删除，避免用过期 split 评测
        os.remove(test_bin)
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
        raise ValueError(
            f"{path} 顶层不是 JSON 对象（实际为 {type(meta).__name__}）")
    return meta


def load_tokenizer(data_dir: str):
    meta = load_meta(data_dir)
    kind = meta.get("tokenizer")
    if not isinstance(kind, str) or not kind:
        raise ValueError(
            f"{os.path.join(data_dir, 'meta.json')} 缺少合法的 tokenizer 字段")
    if kind == "gpt2_bpe":
        return get_gpt2_tokenizer()
    itos = meta.get("itos")
    if not isinstance(itos, list) or not itos or not all(
            isinstance(ch, str) for ch in itos):
        raise ValueError(
            f"{os.path.join(data_dir, 'meta.json')} 的 itos 不是非空字符串数组")
    # 新缓存显式携带 <unk>（char_unk=True）；旧缓存没有该标记，按原编码方式
    # 重建（未知字符→0），保证与旧 bin 的 id 一一对应。
    return CharTokenizer(
        itos, add_unk=config_bool(meta.get("char_unk", False), "char_unk"))


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
        raise ValueError(
            f"split 必须是 'train'/'val'/'test' 之一，收到 {split!r}")
    if split == "test" and allow_test is not True:
        # 只接受真正的 bool True：字符串 "false"、1 等真值若按 truthiness
        # 放行，会把“显式开锁”的闸门变成任何非空值都能绕过。
        raise PermissionError(
            "测试集已尘封：load_bin(split='test') 需显式 allow_test=True"
            "（不接受非布尔真值）；test 只用于最终一次性评估"
            "（lega.evaluate 会写访问账本），训练/调参/选型不得读取")
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"{path} 不存在或为空，请先运行 prepare_bins / prepare_data.py")
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
                    "bin 被替换/篡改/损坏，请重新运行 prepare_bins 重建")
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
    后者把 memmap 的花式索引进 pinned 输出缓冲一步完成（np.take 的 out= 参数）；
    CPU 目标同样用 out= 一步 gather 成 int64。
    """
    global _H2D_STAGE_BYTES
    is_span = isinstance(src, tuple)
    shape = src[1].shape if is_span else src.shape
    dtype = np.uint16 if is_span else src.dtype
    if not device.startswith("cuda"):
        if is_span:
            # 花式索引直接 gather 进 int64 缓冲：np.take(out=...) 在收集时同步
            # 完成 uint16→int64 升位。
            arr = np.empty(shape, dtype=np.int64)
            np.take(src[0], src[1], out=arr)
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
        if (len(_H2D_STAGE) < 32
                and _H2D_STAGE_BYTES + slot_bytes <= _H2D_STAGE_MAX_BYTES):
            slots = [(torch.empty(shape, dtype=tdtype).pin_memory(),
                      torch.cuda.Event()) for _ in range(2)]
            _H2D_STAGE[key] = slots
            _H2D_TURN[key] = 0
            _H2D_STAGE_BYTES += slot_bytes
        else:
            arr = np.take(src[0], src[1]) if is_span else src
            out = torch.from_numpy(np.ascontiguousarray(arr)).to(
                device, non_blocking=True)
            return out if out.dtype == torch.int64 else out.long()
    turn = _H2D_TURN[key]
    buf, ev = slots[turn]
    _H2D_TURN[key] = 1 - turn
    ev.synchronize()                # 等待该槽上一轮的 H2D 拷贝读完 buf
    if is_span:
        # memmap 花式索引进 pinned 缓冲的 np 视图
        np.take(src[0], src[1], out=buf.numpy())
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


def get_train_batch(train: np.memmap, ctx: int, batch_size: int, device: str,
                    rng: np.random.Generator, order_hasher=None):
    """随机偏移采样 (x, y)，y 为 x 右移一位。

    order_hasher（可选）接收本批窗口起点序列的规范化字节流。训练脚本用它
    记录**实际采样顺序**的 SHA-256：仅比较 seed/config 不足以证明各变体
    真的消费了同一批训练数据（例如采样代码路径被改、RNG 被提前消耗），
    而数据顺序不同会让“方法对比”混入数据混杂。
    """
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


def get_val_batch(val: np.memmap, ctx: int, idx: int, device: str):
    """顺序不重叠窗口，idx 为窗口序号（循环取用）；等价于窗口数为 1 的批量版。"""
    return get_val_windows(val, ctx, (idx,), device)


def get_val_windows(val: np.memmap, ctx: int, indices, device: str):
    """一次取多个不重叠窗口拼成批量 (n, ctx)。"""
    n_windows = (len(val) - 1) // ctx
    if n_windows <= 0:
        raise ValueError(f"验证集只有 {len(val)} tokens，不足以组成 ctx={ctx} 的窗口")
    # indices 常已是 np 数组切片：asarray 直接转 int64
    starts = (np.asarray(indices, dtype=np.int64) % n_windows) * ctx
    span = _make_span(starts, ctx)
    # (val, span) 元组：memmap 花式索引直写 pinned 缓冲，设备端升 long 并切 x/y
    both = _stage_to_device((val, span), device)
    return both[:, :-1], both[:, 1:]
