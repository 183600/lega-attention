#!/usr/bin/env python3
"""下载并准备 WikiText-103（raw）数据，编码为 token bin。

来源优先级:
  1. HuggingFace parquet（需 pyarrow）—— Salesforce/wikitext, wikitext-103-raw-v1
  2. Smerity 镜像 zip（zipfile 标准库即可）
  3. rows API（慢，仅适合 --max_tokens 很小的场合）
  4. --local_txt 直接使用本地文本：按输入顺序做“时序式”划分（不是对
     官方 WikiText split 的声明）——
     最早 80% 训练、其后 10% 验证、最新 10% 测试；切点对齐文档边界
     （标题行 > 空行段落 > 换行），绝不随机打乱，保证未来信息不穿越。
     val/test 不参与任何拟合；分词词表与去污统计只依据 train。

来源 1-3 使用 WikiText-103 官方 split：只保证按文档划分/不跨 split，
  不是按时间切分，不能据此宣称时间外推或未来数据证据。

所有来源都会经过统一入口 lega.data.prepare_bins：
  文本坏字符清洗 → 与 val/test 重复的训练行去污染 → 词表只在 train 拟合 →
  val/test transform → token 流硬校验 → 写 data_report.json（防泄漏自检）。

同时写 source_manifest.json：记录原始 URL/revision/文件大小/SHA-256 与清洗后
split 文本哈希；重跑时原始文件变化默认 fail closed，须 --allow_source_change。
HF 默认 revision=main 是可变引用，正式实验请用 --hf_revision 固定 commit。

用法:
  python scripts/prepare_data.py --out_dir data/wikitext103            # 全量
  python scripts/prepare_data.py --out_dir data/wikitext103 --max_tokens 20000000
  python scripts/prepare_data.py --out_dir data/local --local_txt my.txt
"""
import argparse
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF_REPO = "Salesforce/wikitext"
HF_CONFIG = "wikitext-103-raw-v1"
HF_DEFAULT_REVISION = "main"
HF_FILES = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "val": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}
ZIP_URL = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
ROWS_API = ("https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext"
            "&config=wikitext-103-raw-v1&split={split}&offset={offset}&length=100")


class SourceChangedError(RuntimeError):
    """已有原始文件的哈希与上次来源清单不一致（默认 fail closed）。"""


def _hf_base(revision: str) -> str:
    return (f"https://huggingface.co/datasets/{HF_REPO}/resolve/"
            f"{revision}/{HF_CONFIG}")


def _download(url: str, dst: str, desc: str, retries: int = 3,
              expected_sha256: str = None, allow_source_change: bool = False,
              role: str = None, data_dir: str = None) -> dict:
    """下载/复用原始文件并返回可溯源的 file entry。

    已有 dst 默认按上次 manifest 的 SHA-256 校验；不一致时 fail closed，
    只有显式 allow_source_change 才复用并标记 changed/previous_sha256。
    下载完成后同样计算哈希，供本次 manifest 记录。
    """
    import time
    import urllib.request
    from lega.datacheck import file_entry

    changed = False
    previous_sha256 = None
    if os.path.exists(dst):
        actual = file_entry(dst, role or desc, data_dir=data_dir)["sha256"]
        if expected_sha256 is not None and actual != expected_sha256:
            if not allow_source_change:
                raise SourceChangedError(
                    f"原始文件 {desc} 的 SHA-256 与上次来源清单不一致："
                    f"记录 {expected_sha256}，当前 {actual}。上游数据可能已被"
                    "替换/更新；请核对来源，确认后加 --allow_source_change "
                    "显式接受并把变更写入 source_manifest.json。")
            changed = True
            previous_sha256 = expected_sha256
        print(f"[data] 复用已下载 {desc}: {dst}")
        entry = file_entry(dst, role or desc, data_dir=data_dir)
        if changed:
            entry["changed"] = True
            entry["previous_sha256"] = previous_sha256
        return entry

    print(f"[data] 下载 {desc}: {url}")
    tmp = dst + ".part"

    def rep(n, bs, total):
        # n 是已下载块数，bs 是块大小：每 20 块打印一次已下载字节数
        if total > 0 and n % 20 == 0:
            print(f"\r  {desc}: {n * bs / 1e6:.0f}MB / {total / 1e6:.0f}MB", end="", flush=True)
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=rep)
            print()
            os.replace(tmp, dst)
            return file_entry(dst, role or desc, data_dir=data_dir)
        except Exception as e:
            print(f"\n  第 {attempt} 次下载失败: {e}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def from_parquet(out_dir: str, txt_paths: dict, max_tokens: dict,
                 revision: str = HF_DEFAULT_REVISION,
                 expected_hashes: dict = None, allow_source_change: bool = False,
                 manifest_files: list = None, manifest_urls: list = None) -> dict:
    """下载 parquet 并逐文件逐 batch 流式解析、清洗后直接写入各 split 的 txt。

    返回 split -> 写入字符数。manifest_files/manifest_urls 若提供，会追加
    原始文件 entry 与下载 URL，供 source_manifest.json 使用。
    """
    import pyarrow.parquet as pq
    expected_hashes = expected_hashes or {}
    counts = {}
    base = _hf_base(revision)
    for split, files in HF_FILES.items():
        def _iter_lines(files=files):
            for fn in files:
                dst = os.path.join(out_dir, fn)
                url = f"{base}/{fn}"
                entry = _download(
                    url, dst, fn, expected_sha256=expected_hashes.get(fn),
                    allow_source_change=allow_source_change,
                    role=f"hf_parquet:{split}", data_dir=out_dir)
                if manifest_files is not None:
                    manifest_files.append(entry)
                if manifest_urls is not None:
                    manifest_urls.append({"name": fn, "url": url})
                print(f"[data] 解析 {fn}")
                # 按 row group 分批读入，单批行数有界
                pf = pq.ParquetFile(dst)
                for batch in pf.iter_batches(columns=["text"],
                                             batch_size=100_000):
                    yield from batch.column("text").to_pylist()

        lines = _iter_lines()
        try:
            counts[split] = write_clean_lines(lines, txt_paths[split],
                                              max_tokens[split])
        finally:
            lines.close()
    return counts


def from_zip(out_dir: str, txt_paths: dict, max_tokens: dict,
             expected_hashes: dict = None, allow_source_change: bool = False,
             manifest_files: list = None, manifest_urls: list = None) -> dict:
    dst = os.path.join(out_dir, "wikitext-103-raw-v1.zip")
    expected_hashes = expected_hashes or {}
    entry = _download(
        ZIP_URL, dst, "wikitext zip",
        expected_sha256=expected_hashes.get(os.path.basename(dst)),
        allow_source_change=allow_source_change, role="wikitext_zip",
        data_dir=out_dir)
    if manifest_files is not None:
        manifest_files.append(entry)
    if manifest_urls is not None:
        manifest_urls.append({"name": os.path.basename(dst), "url": ZIP_URL})
    counts = {}
    with zipfile.ZipFile(dst) as z:
        names = z.namelist()
        # 该 zip 内为 wikitext-103-raw/wiki.{train,valid,test}.raw；顶层目录名
        # 随镜像可能变化，按文件名后缀定位而不写死完整路径。
        for split, base in [("train", "wiki.train.raw"),
                            ("val", "wiki.valid.raw"),
                            ("test", "wiki.test.raw")]:
            cands = [n for n in names if n == base or n.endswith("/" + base)]
            if not cands:
                raise FileNotFoundError(
                    f"zip 中缺少 {base}；现有条目示例: {names[:8]}")
            with z.open(cands[0]) as f:
                # TextIOWrapper 已做换行翻译，行内容与 split("\n") 一致
                # （空行/行尾空白由 write_clean_lines 统一处理）。
                counts[split] = write_clean_lines(
                    io.TextIOWrapper(f, encoding="utf-8"),
                    txt_paths[split], max_tokens[split])
    return counts


def from_rows_api(max_rows: int, retries: int = 4) -> dict:
    import json as _json
    import time
    import urllib.request
    out = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        offset = 0
        while offset < max_rows:
            url = ROWS_API.format(split="train" if split == "train" else split,
                                  offset=offset)
            url = url.replace("split=val", "split=validation")
            rows = None
            for attempt in range(1, retries + 1):
                try:
                    with urllib.request.urlopen(url, timeout=30) as r:
                        data = _json.load(r)
                    rows = data.get("rows", [])
                    break
                except Exception as e:
                    print(f"\n  rows api 第 {attempt} 次失败({e})")
                    if attempt == retries:
                        raise
                    time.sleep(5 * attempt)
            if not rows:
                break
            out[split].extend(r["row"]["text"] for r in rows)
            offset += len(rows)
            print(f"\r  rows api {split}: {offset} 行", end="", flush=True)
        print()
    return out


def write_clean_lines(lines, dst: str, max_tokens: int) -> int:
    """逐行清洗并流式写入 dst，返回写入的字符数。

    清洗口径：行尾 rstrip、跳过空行内容，但**保留至多一个空行作为块分隔**。
    空行是段落/文档边界；全部删掉会让下游块级去污染与边界审计失去结构。
    max_tokens > 0 时按 1 token ≈ 3.6 字符、留 20% 余量截断。
    截断只在**完整行边界**发生：写入半行会制造被截断的标签/文档，并可能让
    同一段落的残片跨 split。边读边写，写入临时文件后原子替换。

    max_tokens 是数据规模身份：bool/小数/下划线数字串经 int()/乘法会变成
    另一个预算（True→1、1.5→1、float("1_0")→10），却仍报告“截断成功”；
    入口先严格整数化，负值显式拒绝。
    """
    from lega.utils import strict_int
    max_tokens = strict_int(max_tokens, "max_tokens")
    if max_tokens < 0:
        # 负上限会立刻截断第一行，产出三个空 split，却仍可能被后续
        # “无重叠”检查判为通过；必须在写入前显式拒绝。
        raise ValueError(f"max_tokens 不能为负，收到 {max_tokens!r}")
    # 3.6 字符/token × 1.2 余量 = 4.32；用整数除法而不是 `int(max_tokens *
    # 3.6 * 1.2)`：后者在合法但极大的整数预算（如 10**400）上会因 int→float
    # 转换溢出而抛 OverflowError，预算本应只是“不截断”却让整个数据准备失败。
    limit = max_tokens * 432 // 100 if max_tokens else None
    tmp = dst + ".part"
    used, wrote, n_parts = 0, 0, 0
    pending_blank = False
    with open(tmp, "w", encoding="utf-8") as f:
        for raw in lines:
            ln = raw.rstrip()
            if not ln.strip():
                if n_parts:
                    pending_blank = True
                continue
            sep = 0 if not n_parts else (2 if pending_blank else 1)
            if limit is not None and used + sep + len(ln) > limit:
                # 不写半行：宁可提前结束，也不把截断残片当作完整样本。
                break
            if sep:
                f.write("\n" * sep)
            f.write(ln)
            used += sep + len(ln)
            wrote += sep + len(ln)
            n_parts += 1
            pending_blank = False
    os.replace(tmp, dst)
    return wrote


def verify_splits_disjoint(txt_paths: dict, min_chars: int = 40) -> dict:
    """兼容包装：实现已移至 lega.datacheck.verify_splits_disjoint。

    split_protocol 里 "document_disjoint(no_shuffle)" 是上游来源的**声明**，
    不能只靠断言：这里对实际写出的文本做一次可复核的检查——若任意两个
    split 存在相同长行（逐字符相同或 NFKC/空白归一化后相同），说明三个文件
    并非真正互斥，"无共享样本" 的协议被证伪。下游 prepare_bins 也会做去污染
    诊断，这里只是把元数据声明与其实际内容绑定，避免报告"声明一套、数据一套"。
    """
    from lega.datacheck import verify_splits_disjoint as _impl
    return _impl(txt_paths, min_chars=min_chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/wikitext103")
    ap.add_argument("--max_tokens", default=0,
                    help=">0 时截断训练 token 数")
    ap.add_argument("--val_tokens", default=0,
                    help=">0 时截断验证 token 数")
    ap.add_argument("--test_tokens", default=0,
                    help=">0 时截断测试 token 数")
    ap.add_argument("--source", default="auto",
                    choices=["auto", "parquet", "zip", "rows", "local"])
    ap.add_argument("--local_txt", default=None, help="本地 txt 路径（--source local）")
    ap.add_argument("--hf_revision", default=HF_DEFAULT_REVISION,
                    help="HuggingFace 数据集 revision（默认 main 是可变引用；"
                         "正式实验应固定到 commit hash）")
    ap.add_argument("--allow_source_change", action="store_true",
                    help="显式接受原始输入文件与上次 source_manifest.json "
                         "不一致（默认 fail closed）")
    ap.add_argument("--tokenizer", default="gpt2", choices=["gpt2", "auto"])
    ap.add_argument("--near_dup_threshold", default="0.80",
                    help="近似重复阈值（(0,1]；1.0 近似于只保留精确去重）；"
                         "协议参数会写入 data_report")
    ap.add_argument("--near_dup_min_chars", default="200",
                    help="参与近似重复扫描的最小块字符数（>=1）")
    ap.add_argument("--force", action="store_true",
                    help="强制重建 token bin；不传时复用与当前 txt 一致的缓存")
    ap.add_argument("--allow_leaky_split", action="store_true",
                    help="显式允许写出群组跨 split / val-test 重叠的划分"
                         "（仅用于复现或诊断；训练前数据协议闸门仍会拦截）")
    args = ap.parse_args()
    # argparse type=int 会接受 "1_0"（Python 下划线数字串）并静默变成 10，
    # 与 write_clean_lines 内部的严格整数口径冲突；保留原始字符串后统一校验。
    from lega.utils import strict_int

    def _budget(raw, label):
        try:
            value = strict_int(raw, label)
        except (TypeError, ValueError) as e:
            ap.error(str(e))
        if value < 0:
            raise SystemExit(
                f"[data] --max_tokens/--val_tokens/--test_tokens 不能为负，"
                f"非法项：{label}")
        return value

    max_tokens = {
        "train": _budget(args.max_tokens, "--max_tokens"),
        "val": _budget(args.val_tokens, "--val_tokens"),
        "test": _budget(args.test_tokens, "--test_tokens"),
    }
    try:
        near_dup_threshold = float(args.near_dup_threshold)
    except (TypeError, ValueError, OverflowError):
        ap.error("--near_dup_threshold 必须是 (0,1] 内的数值")
    if not (near_dup_threshold > 0.0 and near_dup_threshold <= 1.0):
        ap.error("--near_dup_threshold 必须在 (0,1] 内")
    try:
        near_dup_min_chars = strict_int(args.near_dup_min_chars,
                                        "--near_dup_min_chars")
    except (TypeError, ValueError) as e:
        ap.error(str(e))
    if near_dup_min_chars < 1:
        ap.error("--near_dup_min_chars 必须 >= 1")
    if not isinstance(args.hf_revision, str) or not args.hf_revision.strip():
        ap.error("--hf_revision 必须是非空字符串")

    os.makedirs(args.out_dir, exist_ok=True)
    txt_paths = {"train": os.path.join(args.out_dir, "train.txt"),
                 "val": os.path.join(args.out_dir, "val.txt"),
                 "test": os.path.join(args.out_dir, "test.txt")}

    from lega.datacheck import (build_source_provenance, file_entry,
                                load_source_manifest, write_source_manifest)
    previous_manifest = load_source_manifest(args.out_dir)
    previous_digest = None
    previous_raw_hashes = {}
    if isinstance(previous_manifest, dict):
        digest = previous_manifest.get("digest")
        previous_digest = digest if isinstance(digest, str) else None
        for entry in previous_manifest.get("files") or []:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "")
            sha = entry.get("sha256")
            path = entry.get("path")
            if (role.startswith("hf_parquet:") or role == "wikitext_zip") \
                    and isinstance(sha, str) and isinstance(path, str):
                previous_raw_hashes[os.path.basename(path)] = sha
    raw_files, raw_urls, api_info = [], [], None
    source_path = None
    if args.hf_revision == HF_DEFAULT_REVISION and args.source in (
            "auto", "parquet"):
        print("[data] 警告：--hf_revision=main 是可变引用；正式实验请固定 commit "
              "hash 以便复现（来源仍会记录本次实际使用的 revision）")

    source_used = args.source
    split_protocol = None
    if args.source == "local" or args.local_txt:
        assert args.local_txt, "--source local 需要 --local_txt"
        source_path = os.path.abspath(args.local_txt)
        local_entry = file_entry(args.local_txt, "local_source",
                                 data_dir=args.out_dir)
        previous_local_sha = None
        if (isinstance(previous_manifest, dict)
                and previous_manifest.get("source_path") == source_path):
            for entry in previous_manifest.get("files") or []:
                if (isinstance(entry, dict)
                        and entry.get("role") == "local_source"
                        and isinstance(entry.get("sha256"), str)):
                    previous_local_sha = entry["sha256"]
                    break
        if (previous_local_sha is not None
                and local_entry["sha256"] != previous_local_sha):
            if not args.allow_source_change:
                raise SourceChangedError(
                    f"本地源文件 {source_path} 的 SHA-256 与上次来源清单不一致："
                    f"记录 {previous_local_sha}，当前 {local_entry['sha256']}。"
                    "若这是有意的数据更新，请加 --allow_source_change 显式接受"
                    "并把变更写入 source_manifest.json。")
            local_entry["changed"] = True
            local_entry["previous_sha256"] = previous_local_sha
        raw_files.append(local_entry)
        with open(args.local_txt, "r", encoding="utf-8") as f:
            text = f.read()
        # 时序划分：train=最早、val=其后、test=最新；切点对齐文档边界，
        # 绝不随机打乱。preprocess 级别的清洗/去污/拟合在 prepare_bins 中完成。
        from lega.datacheck import temporal_split_text
        train_text, val_text, test_text, split_info = temporal_split_text(text)
        # 三个切片已各持一份内容，原串此后不再使用
        del text
        split_info["mode"] = "local_temporal"
        split_protocol = split_info
        source_used = "local"
        api_info = {"kind": "local_temporal",
                    "split": split_info.get("mode")}
        leaky_reasons = []
        if split_info["fallback_char_cut"]:
            leaky_reasons.append(
                "本地文本缺少可用换行/文档边界，退化为字符切分，"
                "同一文档/段落可能跨 split")
        elif split_info.get("documents_cross_cut"):
            leaky_reasons.append(
                "切点未落在标题文档边界（标题间距大于划分区间），"
                "有文档跨 split（群组泄露）")
        elif split_info.get("paragraphs_cross_cut"):
            leaky_reasons.append(
                "切点未落在空行段落边界（段落长度大于划分区间），"
                "有段落跨 split（群组泄露）")
        if leaky_reasons:
            detail = "；".join(leaky_reasons)
            if not args.allow_leaky_split:
                raise SystemExit(
                    f"[data] 拒绝写出泄露划分：{detail}。请调整语料格式/划分"
                    "比例或改用官方 split；如仅用于复现/诊断，可显式加 "
                    "--allow_leaky_split（训练前数据协议闸门仍会拦截）。")
            print(f"[data] 警告：--allow_leaky_split 显式放行泄露划分：{detail}")
        else:
            print(f"[data] 时序划分：标题 {split_info.get('n_headings', 0)} 个、"
                  f"段落边界 {split_info.get('n_paragraph_boundaries', 0)} 个，"
                  f"群组单位 {split_info.get('group_unit', '?')} 不跨 split")
        # 逐行清洗并直接写入 txt
        char_counts = {
            split: write_clean_lines(io.StringIO(t), txt_paths[split],
                                     max_tokens[split])
            for split, t in (("train", train_text), ("val", val_text),
                             ("test", test_text))}
        del train_text, val_text, test_text
    else:
        char_counts = None
        if args.source in ("auto", "parquet"):
            attempt_files, attempt_urls = [], []
            try:
                char_counts = from_parquet(
                    args.out_dir, txt_paths, max_tokens,
                    revision=args.hf_revision,
                    expected_hashes=previous_raw_hashes,
                    allow_source_change=args.allow_source_change,
                    manifest_files=attempt_files, manifest_urls=attempt_urls)
                source_used = "parquet"
                raw_files, raw_urls = attempt_files, attempt_urls
                api_info = {"kind": "hf_datasets_parquet",
                            "dataset": HF_REPO, "config": HF_CONFIG,
                            "revision": args.hf_revision}
            except SourceChangedError:
                raise
            except Exception as e:
                print(f"[data] parquet 路径失败: {e}")
                if args.source == "parquet":
                    raise
        if char_counts is None and args.source in ("auto", "zip"):
            attempt_files, attempt_urls = [], []
            try:
                char_counts = from_zip(
                    args.out_dir, txt_paths, max_tokens,
                    expected_hashes=previous_raw_hashes,
                    allow_source_change=args.allow_source_change,
                    manifest_files=attempt_files, manifest_urls=attempt_urls)
                source_used = "zip"
                raw_files, raw_urls = attempt_files, attempt_urls
                api_info = {"kind": "wikitext_zip"}
            except SourceChangedError:
                raise
            except Exception as e:
                print(f"[data] zip 镜像失败: {e}")
                if args.source == "zip":
                    raise
        if char_counts is None and args.source in ("auto", "rows"):
            # 必须使用已严格校验的整数预算，而不是 argparse 原始值：
            # 命令行传入 --max_tokens 20 时 args.max_tokens 仍是字符串，
            # 既不会等于 0，也不能参与 // 运算，rows 兜底路径会直接 TypeError。
            if max_tokens["train"] == 0:
                print("[data] rows API 太慢，需要 --max_tokens 限制；跳过")
            else:
                # token 数 -> 行数估算（平均 ~30 token/行），并设上下限：
                # 下限 500 保证能抓到东西，上限 20000 防止兜底路径发起海量请求
                est_rows = min(max(max_tokens["train"] // 30, 500), 20000)
                lines = from_rows_api(max_rows=est_rows)
                source_used = "rows_api"
                api_info = {"kind": "hf_rows_api", "dataset": HF_REPO,
                            "config": HF_CONFIG, "max_rows": est_rows,
                            "note": "rows API 无 revision 参数，响应本身未逐行"
                                    "留存原始哈希；以清洗后文本哈希为准"}
                char_counts = {
                    split: write_clean_lines(lines.get(split) or [],
                                             txt_paths[split],
                                             max_tokens[split])
                    for split in ("train", "val", "test")}
                del lines
        if char_counts is None:
            raise RuntimeError("所有下载路径均失败，请检查网络或使用 --local_txt")
        # 官方 test split 作为 held-out：checkpoint 选择用 val，最终 PPL 报告用 test，
        # 避免「在报告的指标上做模型选择」的乐观偏差。
        split_protocol = {
            "mode": "official_wikitext103",
            "order": "document_disjoint_official_split(no_shuffle)",
            "shuffled": False,
        }

    print(f"[data] train.txt {char_counts['train']:,} 字符 / "
          f"val.txt {char_counts['val']:,} 字符 / "
          f"test.txt {char_counts['test']:,} 字符")

    # 把划分协议声明与实际写出的文本绑定：官方 split 的 "document_disjoint"
    # 是上游声明，时序划分的 "documents_cross_cut" 是切点检查；这里再对三个
    # 文件做一次跨 split 长行核对，把"声明"升级为"已核对"（或如实标为证伪）。
    if split_protocol is None:
        split_protocol = {}
    split_protocol.update(verify_splits_disjoint(txt_paths))
    if split_protocol.get("missing_files"):
        print(f"[data] 警告：以下 split 文件缺失，无法核对跨 split 长行："
              f"{split_protocol['missing_files']}")
    if not split_protocol.get("verified_disjoint", True):
        ov = {k: v for k, v in split_protocol["overlap_long_lines"].items()
              if v["exact_lines"] or v["normalized_lines"]}
        if ov:
            print(f"[data] 警告：三个 split 的实际文本存在跨 split 共享长行 {ov}——"
                  "与“无共享样本/document_disjoint”声明不符，须核查来源或划分；"
                  "下游去污染会移除 train 侧重复行，但 val/test 重叠只能靠重建划分消除")
    short_ov = split_protocol.get("overlap_short_lines")
    if isinstance(short_ov, dict):
        short_ov = {k: v for k, v in short_ov.items()
                    if isinstance(v, dict)
                    and (v.get("exact_lines") or v.get("normalized_lines"))}
    else:
        short_ov = {}
    if short_ov:
        print(f"[data] 警告：三个 split 存在跨 split 共享短行 {short_ov}"
              f"（<{split_protocol.get('min_chars', '?')} 字符，多为标题/固定用语）；"
              "未自动删除，但重复样本会削弱 held-out 独立性，须人工核查")

    # val/test 之间没有“只改 train”的去污染出口：一旦共享长行，val 选型会
    # 部分预演 test，test 不再是独立 held-out。train 与 val/test 的重叠可由
    # prepare_bins 从 train 侧删除，因此这里只对 val-test 强制 fail closed。
    overlaps = split_protocol.get("overlap_long_lines")
    val_test_overlap = {}
    if isinstance(overlaps, dict):
        pair = overlaps.get("val_test")
        if isinstance(pair, dict):
            exact = pair.get("exact_lines") or 0
            norm = pair.get("normalized_lines") or 0
            if exact or norm:
                val_test_overlap = {"exact_lines": exact,
                                    "normalized_lines": norm}
    if val_test_overlap:
        detail = (f"val/test 共享长行：精确 {val_test_overlap['exact_lines']} 条、"
                  f"空白归一化后 {val_test_overlap['normalized_lines']} 条")
        if not args.allow_leaky_split:
            # 三个 txt 已经落盘；拒绝时必须删除本次写出的泄露划分，否则
            # “命令失败但文件仍在”会让后续手动流程误用这份数据。
            for path in txt_paths.values():
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            raise SystemExit(
                f"[data] 拒绝写出泄露划分：{detail}。val 用于选 checkpoint、"
                "test 用于最终报告，两者共享样本会使 test 不再独立；请核查"
                "来源/重建划分；如仅用于复现/诊断，可显式加 "
                "--allow_leaky_split（训练前数据协议闸门仍会拦截）。")
        print(f"[data] 警告：--allow_leaky_split 显式放行 val/test 重叠：{detail}")

    from lega.data import prepare_bins
    # 来源清单同时覆盖原始下载/本地输入与清洗后的 split 文本：前者回答
    # “字节从哪来”，后者把报告里的文本哈希与原始文件绑定。文本在
    # prepare_bins 之前已落盘且不会再被改写（去污染只改 bin，不改 txt）。
    text_files = [file_entry(txt_paths[split], "text_split",
                             data_dir=args.out_dir)
                  for split in ("train", "val", "test")]
    changes = []
    for entry in raw_files:
        if entry.get("changed"):
            changes.append({
                "role": entry.get("role"),
                "path": entry.get("path"),
                "previous_sha256": entry.get("previous_sha256"),
                "sha256": entry.get("sha256"),
            })
    if changes:
        print(f"[data] 警告：--allow_source_change 显式接受 "
              f"{len(changes)} 个原始文件变化：{[c['path'] for c in changes]}")
    revision = args.hf_revision if source_used == "parquet" else None
    source_provenance = build_source_provenance(
        source_used, raw_files + text_files, revision=revision,
        source_path=source_path, urls=raw_urls, api=api_info,
        changes=changes, previous_digest=previous_digest)
    # force 透传：不传时 prepare_bins 依据源 txt 哈希/协议版本校验缓存，
    # 一致则复用、变化则自动重建；传 --force 则无条件重建。
    prepare_bins(args.out_dir, kind=args.tokenizer, force=args.force,
                 extra_report={"source": source_used,
                               "split_protocol": split_protocol,
                               "source_provenance": source_provenance},
                 near_dup_threshold=near_dup_threshold,
                 near_dup_min_chars=near_dup_min_chars)
    write_source_manifest(args.out_dir, source_provenance)
    print(f"[data] 来源清单已写入 {os.path.join(args.out_dir, 'source_manifest.json')}"
          f"（digest {source_provenance['digest'][:12]}…）")
    print("[data] WikiText-103 准备完成")


if __name__ == "__main__":
    try:
        main()
    except SourceChangedError as e:
        raise SystemExit(f"[data] {e}")
