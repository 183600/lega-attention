"""数据管线：分词器（GPT-2 BPE，tiktoken 不可用时退化为字符级）+ bin 缓存 + 批采样。

约定 data_dir 内文件：
  train.txt / val.txt     原始文本（prepare_data.py 或 make_dummy_data.py 生成）
  train.bin / val.bin     uint16 token 流（prepare_bins 生成）
  meta.json               {"tokenizer": "gpt2_bpe"|"char", "vocab_size": int, ...}
"""
import json
import os
from typing import List

import numpy as np
import torch

_GPT2_CACHE = {}


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
    """字符级分词（离线兜底）。未知字符映射到 0。"""

    def __init__(self, itos: List[str]):
        self.itos = list(itos)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def name(self) -> str:
        return "char"

    @property
    def n_vocab(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def get_gpt2_tokenizer() -> GPT2Tokenizer:
    if "tok" not in _GPT2_CACHE:
        import tiktoken
        _GPT2_CACHE["tok"] = GPT2Tokenizer(tiktoken.get_encoding("gpt2"))
    return _GPT2_CACHE["tok"]


def get_tokenizer(kind: str, train_text: str = "", val_text: str = ""):
    """kind ∈ {gpt2, auto}；gpt2 失败时 auto 退化为字符级，gpt2 直接抛错。"""
    if kind in ("gpt2", "auto"):
        try:
            return get_gpt2_tokenizer()
        except Exception as e:  # 无网络且无本地缓存
            if kind == "gpt2":
                raise
            print(f"[data] tiktoken 不可用({e})，退化为字符级分词")
    chars = sorted(set(train_text) | set(val_text))
    if not chars:
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,\n")
    return CharTokenizer(chars)


# ---------------------------------------------------------------- bin 缓存

def prepare_bins(data_dir: str, kind: str = "auto", force: bool = False) -> dict:
    """把 train.txt/val.txt 编码为 uint16 bin（已存在则跳过，force=True 重建）。"""
    meta_path = os.path.join(data_dir, "meta.json")
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    if (os.path.exists(meta_path) and os.path.exists(train_bin)
            and os.path.exists(val_bin) and not force):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 缓存的分词器与本次请求的 kind 不一致时必须重建，
        # 否则会静默沿用旧分词方式产生的 bin。
        cached_kind = meta.get("requested_kind")
        if cached_kind is None:
            # 旧版 meta 无 requested_kind：按实际分词器推断，
            # 避免 char 缓存被 gpt2 请求静默复用（反之亦然）。
            cached_kind = "gpt2" if meta.get("tokenizer") == "gpt2_bpe" else "char"
        train_valid = os.path.getsize(train_bin) == int(meta.get("train_tokens", -1)) * 2
        val_valid = os.path.getsize(val_bin) == int(meta.get("val_tokens", -1)) * 2
        if (cached_kind == kind or kind == "auto") and train_valid and val_valid:
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
        if not (train_valid and val_valid):
            print("[data] 缓存 bin/meta 大小不一致，重建 bin")
        else:
            print(f"[data] 缓存分词方式({cached_kind})与请求({kind})不一致，重建 bin")

    with open(os.path.join(data_dir, "train.txt"), "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(os.path.join(data_dir, "val.txt"), "r", encoding="utf-8") as f:
        val_text = f.read()

    tok = get_tokenizer(kind, train_text, val_text)
    print(f"[data] 分词器: {tok.name}, vocab={tok.n_vocab}")

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
            parts.append(np.asarray(e, dtype=np.uint16))
            pos = end
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)

    tr = encode_stream(train_text)
    va = encode_stream(val_text)
    # 原子替换：重建中断时不会让旧 meta 搭配半成品 bin 继续被当作有效缓存。
    train_tmp, val_tmp, meta_tmp = train_bin + ".tmp", val_bin + ".tmp", meta_path + ".tmp"
    tr.tofile(train_tmp)
    va.tofile(val_tmp)
    meta = {"tokenizer": tok.name, "requested_kind": kind,
            "vocab_size": tok.n_vocab,
            "train_tokens": int(tr.size), "val_tokens": int(va.size)}
    if tok.name == "char":
        meta["itos"] = tok.itos
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(train_tmp, train_bin)
    os.replace(val_tmp, val_bin)
    os.replace(meta_tmp, meta_path)
    print(f"[data] train {tr.size:,} tokens / val {va.size:,} tokens -> {data_dir}")
    return meta


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(data_dir: str):
    meta = load_meta(data_dir)
    if meta["tokenizer"] == "gpt2_bpe":
        return get_gpt2_tokenizer()
    return CharTokenizer(meta["itos"])


def load_bin(data_dir: str, split: str) -> np.memmap:
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"{path} 不存在或为空，请先运行 prepare_bins / prepare_data.py")
    return np.memmap(path, dtype=np.uint16, mode="r")


# ---------------------------------------------------------------- 批采样

# H2D 乒乓暂存：键 (device, shape) -> [(pinned_buf, event), (pinned_buf, event)]。
# 双槽轮换 + event 保证重写槽位前上一次异步拷贝已完成。
_H2D_STAGE: dict = {}
_H2D_TURN: dict = {}


def _stage_to_device(arr: np.ndarray, device: str) -> torch.Tensor:
    # token id 以 uint16 贯穿 CPU 花式索引、pinned 暂存与 H2D 拷贝，
    # 到目标设备后再升 int64（embedding/交叉熵要求整型索引）。
    src = torch.from_numpy(arr)
    if src.dtype != torch.int64:
        src64 = None
    else:
        src64 = src
    if not device.startswith("cuda"):
        return src64 if src64 is not None else src.long()
    key = (device, tuple(arr.shape), src.dtype)
    slots = _H2D_STAGE.get(key)
    if slots is None:
        if len(_H2D_STAGE) < 32:    # 形状种类有限（训练/验证各一两种）
            slots = [(torch.empty(arr.shape, dtype=src.dtype).pin_memory(),
                      torch.cuda.Event()) for _ in range(2)]
            _H2D_STAGE[key] = slots
            _H2D_TURN[key] = 0
        else:
            out = src.to(device, non_blocking=True)
            return out if out.dtype == torch.int64 else out.long()
    turn = _H2D_TURN[key]
    buf, ev = slots[turn]
    _H2D_TURN[key] = 1 - turn
    ev.synchronize()                # 等待该槽上一轮的 H2D 拷贝读完 buf
    buf.copy_(src)
    out = buf.to(device, non_blocking=True)
    ev.record()
    return out if out.dtype == torch.int64 else out.long()


# 按 ctx 缓存索引基底，每个 ctx 只需新建一次 arange
_SPAN_BASE: dict = {}


def _span_base(ctx: int) -> np.ndarray:
    base = _SPAN_BASE.get(ctx)
    if base is None:
        base = np.arange(ctx + 1)
        _SPAN_BASE[ctx] = base
    return base


def get_train_batch(train: np.memmap, ctx: int, batch_size: int, device: str,
                    rng: np.random.Generator):
    """随机偏移采样 (x, y)，y 为 x 右移一位。"""
    if len(train) < ctx + 1:
        raise ValueError(f"训练集只有 {len(train)} tokens，不足以采样 ctx={ctx}")
    ix = rng.integers(0, len(train) - ctx, size=batch_size)
    # 向量化索引：一次性取 (batch_size, ctx+1)
    span = ix[:, None] + _span_base(ctx)
    # 不升 int64：uint16 贯穿到 H2D，设备端再转 long
    buf = np.ascontiguousarray(train[span])         # memmap 花式索引自动拷贝
    # x/y 合并为一次 pinned H2D 拷贝（(B, ctx+1)），到设备后再切片
    both = _stage_to_device(buf, device)
    return both[:, :-1], both[:, 1:]


def get_val_batch(val: np.memmap, ctx: int, idx: int, device: str):
    """顺序不重叠窗口，idx 为窗口序号（循环取用）；等价于窗口数为 1 的批量版。"""
    return get_val_windows(val, ctx, (idx,), device)


def get_val_windows(val: np.memmap, ctx: int, indices, device: str):
    """一次取多个不重叠窗口拼成批量 (n, ctx)，减少验证前向的调用次数。"""
    n_windows = (len(val) - 1) // ctx
    if n_windows <= 0:
        raise ValueError(f"验证集只有 {len(val)} tokens，不足以组成 ctx={ctx} 的窗口")
    # indices 常已是 np 数组切片：asarray 直接转 int64
    starts = (np.asarray(indices, dtype=np.int64) % n_windows) * ctx
    span = starts[:, None] + _span_base(ctx)
    buf = np.ascontiguousarray(val[span])           # (n, ctx+1) uint16，设备端升 long
    both = _stage_to_device(buf, device)            # 单次 H2D，设备上切 x/y
    return both[:, :-1], both[:, 1:]
