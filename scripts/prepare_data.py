#!/usr/bin/env python3
"""下载并准备 WikiText-103（raw）数据，编码为 token bin。

来源优先级:
  1. HuggingFace parquet（需 pyarrow）—— Salesforce/wikitext, wikitext-103-raw-v1
  2. Smerity 镜像 zip（zipfile 标准库即可）
  3. rows API（慢，仅适合 --max_tokens 很小的场合）
  4. --local_txt 直接使用本地文本：按时间先后做叙事性划分——
     最早 80% 训练、其后 10% 验证、最新 10% 测试；切点对齐文档边界
     （标题行 > 空行段落 > 换行），绝不随机打乱，保证未来信息不穿越。
     val/test 不参与任何拟合；分词词表与去污统计只依据 train。

来源 1-3 使用 WikiText-103 官方 split（按文档划分、天然 held-out test）。

所有来源都会经过统一入口 lega.data.prepare_bins：
  文本坏字符清洗 → 与 val/test 重复的训练行去污染 → 词表只在 train 拟合 →
  val/test transform → token 流硬校验 → 写 data_report.json（防泄漏自检）。

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

HF_BASE = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
           "wikitext-103-raw-v1")
HF_FILES = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "val": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}
ZIP_URL = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
ROWS_API = ("https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext"
            "&config=wikitext-103-raw-v1&split={split}&offset={offset}&length=100")


def _download(url: str, dst: str, desc: str, retries: int = 3) -> None:
    import time
    import urllib.request
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
            return
        except Exception as e:
            print(f"\n  第 {attempt} 次下载失败: {e}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def from_parquet(out_dir: str) -> dict:
    import pyarrow.parquet as pq
    texts = {}
    for split, files in HF_FILES.items():
        parts = []
        for fn in files:
            dst = os.path.join(out_dir, fn)
            if not os.path.exists(dst):
                _download(f"{HF_BASE}/{fn}", dst, fn)
            t = pq.read_table(dst, columns=["text"])
            parts.append(t.column("text").to_pylist())
            print(f"[data] 已解析 {fn}: {len(parts[-1]):,} 行")
        texts[split] = parts
    return {split: [line for part in parts for line in part]
            for split, parts in texts.items()}


def from_zip(out_dir: str) -> dict:
    dst = os.path.join(out_dir, "wikitext-103-raw-v1.zip")
    if not os.path.exists(dst):
        _download(ZIP_URL, dst, "wikitext zip")
    texts = {}
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
                texts[split] = io.TextIOWrapper(f, encoding="utf-8").read().split("\n")
    return texts


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


def clean_lines(lines, max_tokens: int, tokenizer_kind: str) -> str:
    if not max_tokens:
        return "\n".join(ln.rstrip() for ln in lines if ln.strip())
    # 粗略按 1 token ≈ 3.6 字符估算，留 20% 余量
    limit = int(max_tokens * 3.6 * 1.2)
    parts = []
    used = 0
    for raw in lines:
        ln = raw.rstrip()
        if not ln.strip():
            continue
        sep = int(bool(parts))
        take = min(len(ln), limit - used - sep)
        if take < 0:
            break
        if sep:
            parts.append("\n")
        parts.append(ln[:take])
        used += sep + take
        if used >= limit:
            break
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/wikitext103")
    ap.add_argument("--max_tokens", type=int, default=0, help=">0 时截断训练 token 数")
    ap.add_argument("--val_tokens", type=int, default=0, help=">0 时截断验证 token 数")
    ap.add_argument("--test_tokens", type=int, default=0, help=">0 时截断测试 token 数")
    ap.add_argument("--source", default="auto",
                    choices=["auto", "parquet", "zip", "rows", "local"])
    ap.add_argument("--local_txt", default=None, help="本地 txt 路径（--source local）")
    ap.add_argument("--tokenizer", default="gpt2", choices=["gpt2", "auto"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_txt = os.path.join(args.out_dir, "train.txt")
    val_txt = os.path.join(args.out_dir, "val.txt")
    test_txt = os.path.join(args.out_dir, "test.txt")

    source_used = args.source
    split_protocol = None
    if args.source == "local" or args.local_txt:
        assert args.local_txt, "--source local 需要 --local_txt"
        text = open(args.local_txt, "r", encoding="utf-8").read()
        # 时序划分：train=最早、val=其后、test=最新；切点对齐文档边界，
        # 绝不随机打乱。preprocess 级别的清洗/去污/拟合在 prepare_bins 中完成。
        from lega.datacheck import temporal_split_text
        train_text, val_text, test_text, split_info = temporal_split_text(text)
        split_info["mode"] = "local_temporal"
        split_protocol = split_info
        source_used = "local"
        if split_info["fallback_char_cut"]:
            print("[data] 警告：本地文本缺少可用换行/文档边界，退化为字符切分；"
                  "同一文档可能跨 split，请检查语料格式")
        elif split_info.get("documents_cross_cut"):
            print("[data] 警告：切点未落在标题文档边界（标题间距大于划分区间），"
                  "有文档跨 split（群组泄露）；请调整语料/划分比例或改用官方 split")
        else:
            print(f"[data] 时序划分：标题 {split_info.get('n_headings', 0)} 个，"
                  "文档不跨 split")
        train_text = clean_lines(train_text.split("\n"), args.max_tokens, args.tokenizer)
        val_text = clean_lines(val_text.split("\n"), args.val_tokens, args.tokenizer)
        test_text = clean_lines(test_text.split("\n"), args.test_tokens, args.tokenizer)
    else:
        lines = None
        if args.source in ("auto", "parquet"):
            try:
                lines = from_parquet(args.out_dir)
                source_used = "parquet"
            except Exception as e:
                print(f"[data] parquet 路径失败: {e}")
                if args.source == "parquet":
                    raise
        if lines is None and args.source in ("auto", "zip"):
            try:
                lines = from_zip(args.out_dir)
                source_used = "zip"
            except Exception as e:
                print(f"[data] zip 镜像失败: {e}")
                if args.source == "zip":
                    raise
        if lines is None and args.source in ("auto", "rows"):
            if args.max_tokens == 0:
                print("[data] rows API 太慢，需要 --max_tokens 限制；跳过")
            else:
                # token 数 -> 行数估算（平均 ~30 token/行），并设上下限：
                # 下限 500 保证能抓到东西，上限 20000 防止兜底路径发起海量请求
                est_rows = min(max(args.max_tokens // 30, 500), 20000)
                lines = from_rows_api(max_rows=est_rows)
                source_used = "rows_api"
        if lines is None:
            raise RuntimeError("所有下载路径均失败，请检查网络或使用 --local_txt")
        train_text = clean_lines(lines["train"], args.max_tokens, args.tokenizer)
        val_text = clean_lines(lines["val"], args.val_tokens, args.tokenizer)
        # 官方 test split 作为 held-out：checkpoint 选择用 val，最终 PPL 报告用 test，
        # 避免「在报告的指标上做模型选择」的乐观偏差。
        test_text = clean_lines(lines["test"], args.test_tokens, args.tokenizer)
        split_protocol = {
            "mode": "official_wikitext103",
            "order": "document_disjoint_official_split(no_shuffle)",
            "shuffled": False,
        }

    with open(train_txt, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write(val_text)
    with open(test_txt, "w", encoding="utf-8") as f:
        f.write(test_text)
    print(f"[data] train.txt {len(train_text):,} 字符 / val.txt {len(val_text):,} 字符"
          f" / test.txt {len(test_text):,} 字符")

    from lega.data import prepare_bins
    prepare_bins(args.out_dir, kind=args.tokenizer, force=True,
                 extra_report={"source": source_used,
                               "split_protocol": split_protocol})
    print("[data] WikiText-103 准备完成")


if __name__ == "__main__":
    main()
