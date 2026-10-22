"""独立困惑度评测：从 checkpoint 计算 PPL（默认 held-out test，缺 test.bin 时回退 val）。

用法:
  python -m lega.evaluate --ckpt out/tiny/lega/model_best.pt --data_dir data/dummy \
      --out results/tiny/eval_lega.json
"""
import argparse
import functools
import json
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from .data import get_val_windows, load_bin, load_tokenizer
from .datacheck import append_test_access
from .model import (gate_extra_params, gate_extra_params_per_head,
                    load_model, count_params)
from .preflight import (allow_protocol_violations,
                        data_protocol_preflight)
from .repro import (load_provenance, sha256_file,
                    test_data_fingerprint as _test_data_fingerprint)
from .train import evaluate_val, _val_eval_indices
from .utils import (amp_autocast, amp_dtype_for, config_bool, enable_tf32,
                    get_device, now_str, save_json, set_seed, strict_int)


# 最差窗口筛查的候选窗口上限：与 PPL 的 eval_iters 各自独立（前者排序、
# 后者求平均），必须显式记录，避免把“64 个候选里的最差”说成全量最差。
WORST_SCREEN_MAX = 64


def repeated_test_checkpoint(test_access, ckpt_sha256,
                             test_data_fingerprint=None) -> bool:
    """该 checkpoint 是否已在同一 test 数据版本上评测过。

    纯函数，供评测主流程与单元测试共用：重复在 test 上评测是“测试集变隐式
    训练集”的直接信号，调用方据此决定拒绝还是放行。传入当前 test 指纹时按
    (checkpoint, test 版本) 精确判定：重建 test.bin 后的评测不是旧版本的
    重复访问，不应误拒绝。
    """
    if not isinstance(test_access, dict) or not test_access or not ckpt_sha256:
        return False
    prefix = str(ckpt_sha256)[:16]
    details = test_access.get("repeated_evaluations")
    if test_data_fingerprint is not None and isinstance(details, list):
        ver = str(test_data_fingerprint)
        for item in details:
            if not isinstance(item, dict):
                continue
            if (item.get("ckpt_prefix") == prefix
                    and item.get("test_data_fingerprint") == ver):
                return True
        return False
    repeated = test_access.get("checkpoints_evaluated_multiple_times") or []
    return prefix in repeated


def worst_windows_allowed(split, worst_on_test: bool) -> bool:
    """是否允许在该 split 上转储最差窗口（坏案例）。

    test 上默认禁止：最差窗口连同上下文/预测暴露 test 内容，交人工迭代等于
    据 test 改进模型。只有显式 worst_on_test 才放行。
    """
    # 只接受真正的 bool True：字符串 "false"/0 等真值若按 truthiness 放行，
    # 会把“test 最差窗口尘封”的闸门变成任何非空值都能绕过。
    return split != "test" or worst_on_test is True


def _preserve_eval_mode(fn):
    """评测辅助函数临时切到 eval 并在结束时恢复调用方模式（dropout 不影响结果）。"""
    @functools.wraps(fn)
    def wrapper(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        try:
            return fn(model, *args, **kwargs)
        finally:
            if was_training:
                model.train()
    return wrapper


def _positive_int_or_none(value):
    """配置里的数值→正整数；bool/非有限浮点/小数/越界值返回 None。

    配置是外部可改的评测协议；把 8.5 截断成 8 或把 True 当成 1 会让评测
    声称使用了与训练配置不同的上下文/窗口数，PPL 因而不可比。
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        value = int(value)
    elif isinstance(value, str):
        # int("1_0")==10：下划线不是规范整数串，不能冒充另一个配置值。
        if re.fullmatch(r"[+-]?\d+", value.strip()) is None:
            return None
        value = int(value)
    else:
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(f) or not f.is_integer():
            return None
        value = int(f)
    return value if value > 0 else None


def checkpoint_selection_info(ck, steps_planned_fallback=None) -> dict:
    """从 checkpoint 元数据还原“被评测模型是在第几步、用哪个 val_loss 选出的”。

    run_all 比较的是各变体的 `model_best.pt`；它们都在 val 上按最小损失选出，
    但选型步可能不同（同一预算内早停/波动）。不记录选型步就无法审计
    “same budget”是否真的可比，也无法发现某个变体在极早步被偶然选中。
    缺失/非法字段一律返回 None（旧 checkpoint 不伪造），计划步数可回退到
    训练配置的 steps。
    """
    ck = ck if isinstance(ck, dict) else {}
    step = _positive_int_or_none(ck.get("step"))
    planned = _positive_int_or_none(ck.get("steps_planned"))
    if planned is None:
        planned = _positive_int_or_none(steps_planned_fallback)
    frac = None
    if step is not None and planned is not None:
        try:
            frac = float(step) / float(planned)
        except (OverflowError, ValueError):
            # 超大整数无法转成有限 float（例如被篡改的 step=10**400）：
            # 选型步本身仍可记录，但不能让比例计算把整个评测打崩。
            frac = None
    val_loss = None
    raw_loss = ck.get("val_loss")
    if (raw_loss is not None
            and not isinstance(raw_loss, (bool, np.bool_))
            and not (isinstance(raw_loss, str) and "_" in raw_loss)):
        try:
            val_loss = float(raw_loss)
        except (TypeError, ValueError, OverflowError):
            val_loss = None
        if val_loss is not None and not math.isfinite(val_loss):
            val_loss = None
    return {
        "ckpt_step": step,
        "steps_planned": planned,
        "selection_step_fraction": frac,
        "selection_step_exceeds_planned": bool(
            step is not None and planned is not None and step > planned),
        "ckpt_val_loss": val_loss,
    }


def worst_window_screening(val_len: int, ctx: int,
                           max_windows: int = WORST_SCREEN_MAX) -> dict:
    """最差窗口筛查范围（候选数/总窗口数/覆盖率），供结果落盘与汇总注脚。"""
    max_windows = strict_int(max_windows, "max_windows")
    if max_windows < 1:
        raise ValueError(f"max_windows 必须 >= 1，收到 {max_windows!r}")
    screened, n_windows = _val_eval_indices(val_len, ctx, max_windows)
    return {"screened": int(len(screened)), "total": int(n_windows),
            "screen_max": max_windows}


@torch.inference_mode()
@_preserve_eval_mode
def collect_worst_windows(model, val, ctx: int, device: str, tok,
                          n_worst: int = 3, amp: bool = False,
                          max_windows: int = WORST_SCREEN_MAX,
                          batch_windows: int = None,
                          split: str = "val") -> list:
    """语言建模的坏案例：挑出下一 token 预测最差的验证窗口并解码。

    以"窗口末位的下一 token 预测"为坏案例单元（与贪心生成评测口径一致）：
    损失 = 末位交叉熵；同时给出模型的 top-5 预测与真实 token，用于人工区分
    "样本本身不可预测（噪声/生僻 token）"与"模型系统性盲区"。
    n_worst<=0（0=关闭）返回空列表，不白跑一遍前向。
    """
    n_worst = strict_int(n_worst, "n_worst")
    if n_worst <= 0:
        return []
    max_windows = strict_int(max_windows, "max_windows")
    if max_windows < 1:
        raise ValueError(f"max_windows 必须 >= 1，收到 {max_windows!r}")
    indices, _ = _val_eval_indices(len(val), ctx, max_windows)
    if batch_windows is None:
        batch_windows = max(1, min(16384 // max(1, ctx), len(indices)))
    else:
        batch_windows = _positive_int_or_none(batch_windows)
        if batch_windows is None:
            raise ValueError(
                f"batch_windows 必须是正整数，收到 {batch_windows!r}")
    amp_ctx = amp_autocast(device, bool(amp))
    rows = []
    for s in range(0, len(indices), batch_windows):
        chunk = indices[s:s + batch_windows]
        x, y = get_val_windows(val, ctx, chunk, device)
        with amp_ctx:
            out = model(x, last_logits_only=True)
        logits = out["logits"][:, -1, :].float()
        gold = y[:, -1]
        # 交叉熵与 top-5 概率共用同一份 log_softmax 结果。
        logp = F.log_softmax(logits, dim=-1)
        losses = -logp.gather(1, gold.unsqueeze(1)).squeeze(1)
        # 小词表（如字符级冒烟数据）不足 5 类时 topk 会直接越界；按实际
        # 词表截断，保证转储仍可用。
        top_v, top_i = torch.topk(logp, min(5, logp.shape[-1]), dim=-1)
        top_p = top_v.exp()
        # 整表转 Python 标量后再构造行记录。
        loss_l, gold_l = losses.tolist(), gold.tolist()
        top_p_l, top_i_l = top_p.tolist(), top_i.tolist()
        x_l = x.tolist()
        for r in range(len(chunk)):
            ctx_text = tok.decode(x_l[r])
            gold_txt = tok.decode([gold_l[r]])
            # gold 在窗口中的所有出现位置（token 级精确匹配）。最差窗口的
            # 上下文尾部只有 160 字符，若 gold 出现在更早位置，人工复核时
            # 看不到证据句，无法区分“标注/构造问题”“检索失败”与“复制失败”。
            # 这里把最后一次出现附近的原文一并落盘，并记录到末尾的距离。
            gold_positions = [i for i, t in enumerate(x_l[r])
                              if t == gold_l[r]]
            gold_pos = gold_positions[-1] if gold_positions else None
            gold_excerpt = None
            if gold_pos is not None:
                lo = max(0, gold_pos - 48)
                hi = min(len(x_l[r]), gold_pos + 49)
                gold_excerpt = tok.decode(x_l[r][lo:hi])
                if len(gold_excerpt) > 240:
                    gold_excerpt = gold_excerpt[:240] + "…"
                gold_excerpt = gold_excerpt.replace("\n", " ⏎ ")
            rows.append({
                "window": int(chunk[r]),
                "split": str(split),
                "loss": float(loss_l[r]),
                "context_tail": ctx_text[-160:],
                # gold 是否出现在完整上下文窗口内。必须在采集时用全窗口判定：
                # context_tail 只有 160 字符（窗口尾部），早先出现过的 gold 若只
                # 看尾部会被误判为"从未出现"（复制/检索失败被错记成预测失败）。
                # 按 token id 精确判定：字节级 BPE 下单 token decode 可能不是
                # 完整字符（非 ASCII 时文本包含判定漏报），文本包含还会把
                # 其他 token 内的子串误报为 gold 出现。
                "gold_in_context": bool(gold_positions),
                "gold_next": gold_txt,
                "gold_occurrences": len(gold_positions),
                "gold_last_pos": (int(gold_pos) if gold_pos is not None else None),
                "gold_distance_tokens": (int(len(x_l[r]) - gold_pos - 1)
                                         if gold_pos is not None else None),
                "gold_context_excerpt": gold_excerpt,
                "pred_top5": [(tok.decode([i]), round(float(p), 4))
                              for p, i in zip(top_p_l[r], top_i_l[r])],
            })
    rows.sort(key=lambda r: -r["loss"])
    return rows[:n_worst]


def json_safe_eval_metrics(metrics: dict) -> dict:
    """把评测指标规范化为 JSON 安全且不误导的字段。

    训练内评测（train.py 的 do_eval）在写入前用 _nan_guard 对非有限 val_loss
    显式中止；独立评测入口没有这道闸门，必须在写盘前显式处理，否则：
      * json.dump 会把 NaN/Inf 序列化为非法 JSON 字面量（NaN/Infinity），
        外部消费者（jq 等）直接解析失败；
      * math.exp(min(20, float("nan"))) 恰为 exp(20)（min 对 NaN 返回第一个
        参数），而 bool(nan > 20) 为 False——失效 checkpoint 会被伪装成
        “正常但很大”的有限 PPL，且不触发任何截断标注。
    非有限值统一记为 None（缺失值语义），由下游按缺失处理；有限值沿用
    exp(min(20, loss)) 与 >20 的截断口径，与 train.py 完全一致。
    """
    metrics = metrics if isinstance(metrics, dict) else {}
    raw_loss = metrics.get("val_loss")
    try:
        # bool 是 int 子类，float(True)=1.0；坏记录不能伪装成 val_loss=1.0
        # （对应一个看似正常的 PPL）。下划线数字串同样不是规范 JSON 数值。
        vl = (None if isinstance(raw_loss, (bool, np.bool_))
              or (isinstance(raw_loss, str) and "_" in raw_loss)
              else float(raw_loss))
    except (TypeError, ValueError, OverflowError):
        vl = None
    if vl is not None and math.isfinite(vl):
        ppl = math.exp(min(20, vl))
        loss, clamped = vl, bool(vl > 20)
    else:
        ppl, loss, clamped = None, None, None
    gmean = metrics.get("gate_mean")
    if gmean is not None:
        if (isinstance(gmean, (bool, np.bool_))
                or (isinstance(gmean, str) and "_" in gmean)):
            gmean = None
        else:
            try:
                gmean = float(gmean)
            except (TypeError, ValueError, OverflowError):
                gmean = None
            else:
                if not math.isfinite(gmean):
                    gmean = None
    return {"val_ppl": ppl, "val_loss": loss, "val_ppl_clamped": clamped,
            "gate_mean": gmean}


def _resolve_eval_amp(cfg_used: dict, device: str) -> bool:
    """按 checkpoint 记录的实际 AMP 精度决定评测是否启用 autocast。

    config_used.json 同时记录请求开关 ``amp`` 与训练时真正生效的
    ``amp_dtype``（bf16 不可用时为 null）。若只重算 ``amp``，在支持 bf16 的
    机器上会把“训练时 fp32 回退”的 checkpoint 用 bf16 评测，PPL/门控指标
    与训练内验证不是同一数值路径。旧 checkpoint 没有 amp_dtype 时保留按
    ``amp`` 重算的兼容行为。
    """
    cfg_used = cfg_used if isinstance(cfg_used, dict) else {}
    if "amp_dtype" in cfg_used:
        raw = cfg_used.get("amp_dtype")
        if raw is None:
            return False
        if not isinstance(raw, str):
            return False
        name = raw.strip().lower().replace("torch.", "")
        if name not in ("bfloat16", "float16"):
            return False
        available = amp_dtype_for(device, True)
        return (available is not None
                and str(available).replace("torch.", "") == name)
    return amp_dtype_for(
        device, config_bool(cfg_used.get("amp", True), "amp")) is not None


def _read_run_eval(run_dir: str) -> dict:
    """读取训练 run 的 eval.json；缺失/损坏/结构异常返回空 dict（不阻塞评测）。"""
    path = os.path.join(run_dir, "eval.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _nonnegative_metric(v):
    """训练统计字段的非负有限数值规范化；坏值返回 None（不伪造吞吐）。"""
    if v is None or isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, str) and "_" in v:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if math.isfinite(v) and v >= 0 else None


def load_train_throughput(run_dir: str):
    """从训练 run 的 eval.json 继承训练过程记录（train_step_time_ms/tokens_per_sec）。

    旧 run 没有该字段时返回 (None, None)，消费方显示 '-' 而不是编造数值。
    文件缺失/损坏同样安全返回 (None, None)，不阻塞评测。
    """
    d = _read_run_eval(run_dir)
    return (_nonnegative_metric(d.get("train_step_time_ms")),
            _nonnegative_metric(d.get("train_tokens_per_sec")))


def load_train_peak_mem_mb(run_dir: str):
    """从训练 run 的 eval.json 继承训练进程统计字段（数值型）。

    必须随评测结果落盘：run_all 的汇总列只读 results/eval_*.json；
    缺失/损坏返回 None。
    """
    # bool/负数/NaN/Inf 都不是真实峰值内存；float(True)=1.0 会让坏记录
    # 在汇总表里冒出一个看似合法的 1 MB。
    return _nonnegative_metric(_read_run_eval(run_dir).get("train_peak_mem_mb"))


def load_train_coverage(run_dir: str):
    """从训练 run 的 eval.json 继承训练数据覆盖率。

    独立评测（test split）也需要知道 checkpoint 训练时看过多少数据：
    epochs_seen<1 的结果不能当作收敛。返回 dict（缺失字段不含该键），
    文件缺失/损坏返回空 dict，不阻塞评测。
    """
    d = _read_run_eval(run_dir)
    out = {}
    for src, dst in (("tokens_seen", "train_tokens_seen"),
                     ("train_tokens", "train_tokens")):
        v = d.get(src)
        if v is None:
            continue
        try:
            v = strict_int(v, f"eval.json.{src}")
        except ValueError:
            continue
        if v >= 0:
            out[dst] = v
    e = d.get("epochs_seen")
    if (e is not None
            and not isinstance(e, (bool, np.bool_))
            and not (isinstance(e, str) and "_" in e)):
        try:
            e = float(e)
            if math.isfinite(e) and e >= 0:
                out["epochs_seen"] = e
        except (TypeError, ValueError, OverflowError):
            pass
    return out


def main():
    p = argparse.ArgumentParser(description="PPL 评测（默认 held-out test，缺 test.bin 时回退 val）")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.add_argument("--eval_iters", default=None,
                   help="评测窗口数；默认继承训练配置，训练配置缺失时为 50"
                        "（与 train.py 的默认值一致，保证窗口规格可比）")
    p.add_argument("--worst_n", type=int, default=0,
                   help="转储预测最差的 N 个验证/测试窗口（语言建模坏案例，0=关闭）")
    p.add_argument("--worst_on_test", action="store_true",
                   help="允许在 test split 上转储最差窗口（坏案例）。默认关闭："
                        "test 只能最终评估一次，把 test 错误清单交给人工迭代会"
                        "让测试集变成隐式训练集。需要审阅 test 错误时显式开启。")
    p.add_argument("--split", default="auto",
                   choices=["auto", "test", "val"],
                   help="默认 auto：存在 test.bin 时用 held-out test（checkpoint 选择在 "
                        "val 上完成，报告指标必须在独立 split 上），否则回退 val")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow_repeat_test", action="store_true",
                   help="允许同一 checkpoint 在 test 上重复评测（尘封协议下这是"
                        "泄露前兆：测试集只能最终一次性评估；默认拒绝，确需重测"
                        "时显式放行，账本仍会留痕）")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 检测到划分/重叠泄露也继续评测（仅用于"
                        "复现历史实验；报告会标记 override，结果不得用于正式结论）")
    args = p.parse_args()
    if args.eval_iters is not None:
        # 保留原始字符串后再 strict_int：argparse 的 type=int 会把 "1_0"
        # 静默解析成 10，绕过“非法数值显式拒绝”的入口约定。
        try:
            args.eval_iters = strict_int(args.eval_iters, "--eval_iters")
        except (TypeError, ValueError) as e:
            p.error(str(e))
        if args.eval_iters <= 0:
            p.error("--eval_iters 必须是正整数")

    # 评测同样不能在已标记泄露的数据上产出“正式指标”：训练闸门只保护
    # checkpoint 生产，独立 evaluate/复现历史 run 也必须先过同一份报告。
    allow_override = allow_protocol_violations(
        args.allow_data_protocol_violations)
    data_preflight = data_protocol_preflight(
        args.data_dir, allow_override=allow_override)
    if not data_preflight["report_present"]:
        print("[eval] 警告：数据目录缺少 data_report.json，"
              "无法执行评测前防泄漏闸门（旧数据或未走统一入口）")
    elif data_preflight["critical_violations"]:
        print(f"[eval] ⚠ 数据协议自检发现 "
              f"{len(data_preflight['critical_violations'])} 个严重问题，"
              "但已显式放行（结果不得用于正式结论）")

    device = get_device(args.device)
    enable_tf32(device)
    model, ck = load_model(args.ckpt, device=device)
    # 评测同样固定随机源（评测本身无随机性，但保证同一 checkpoint 在任何
    # 入口下都从同一 RNG 状态出发）。种子来自 checkpoint 元数据：bool/小数
    # 若被 int() 截断，评测会以与记录不同的种子播种（1.5 → 1、True → 1），
    # 身份审计随之失效——非法元数据必须显式拒绝而不是静默改写。
    seed_raw = ck.get("seed")
    # 输出/播种都用规范化后的整数：源元数据里的 5.0 不能写成 5.0，否则
    # eval.json 与台账的 seed 身份会与真正 set_seed 的值不同。
    seed = (None if seed_raw is None
            else strict_int(seed_raw, "checkpoint seed"))
    set_seed(0 if seed is None else seed)
    split = args.split
    if split == "auto":
        split = "test" if (os.path.exists(os.path.join(args.data_dir, "test.bin"))
                           and os.path.getsize(os.path.join(args.data_dir, "test.bin")) > 0) \
            else "val"
    if split == "val":
        # 与 checkpoint 选择（best on val）共用同一 split：报告指标存在选择偏差，
        # 须明确提示而非静默；正式数据请用 prepare_data.py 生成 held-out test。
        print("[eval] 警告：正在 val split 上评测——这也是 best checkpoint 的选择集，"
              "报告 PPL 偏乐观；可用 prepare_bins 生成 test.bin 后在 held-out test 上评测")
    # checkpoint 内容指纹：test 访问账本与结果 JSON 共用同一值。
    ckpt_sha256 = sha256_file(args.ckpt)
    # 账本必须绑定 test 数据版本：只看 checkpoint 会把“重建 test 后的合法
    # 重测”误判为重复，也无法审计同一模型到底评测过哪些测试集。
    test_data_fingerprint = (None if split != "test"
                             else _test_data_fingerprint(args.data_dir))
    test_access = None
    if split == "test":
        # 测试集尘封协议：每次在 test 上评测都写入访问账本（data_dir 下），
        # 同一 checkpoint 被反复评测会显式报警——反复据 test 调参会把测试集
        # 变成隐式训练集；账本让这种行为可审计。
        # 只读数据目录等无法写账本的情况不阻塞评测，但必须显式告警。
        try:
            test_access = append_test_access(args.data_dir, {
                "time": now_str(),
                "ckpt": os.path.abspath(args.ckpt),
                "ckpt_sha256": ckpt_sha256,
                "variant": ck.get("variant", model.variant),
                "seed": seed,
                "test_data_fingerprint": test_data_fingerprint,
            })
        except OSError as e:
            print(f"[eval] 警告：test 访问账本写入失败（{e}）；"
                  "本次 test 评测未被记录")
        if repeated_test_checkpoint(test_access, ckpt_sha256,
                                    test_data_fingerprint):
            # 同一 checkpoint 重复在 test 上评测是“测试集变隐式训练集”的直接
            # 信号：账本已记录，但仅打印警告不足以阻止泄露循环（如反复
            # run_all --skip_train 据 test 迭代）。默认拒绝，需显式放行。
            print(f"[eval] ⚠ 同一 checkpoint 在 test 上被重复评测"
                  f"（累计 {test_access['evaluations']} 次、"
                  f"{test_access['distinct_checkpoints']} 个 checkpoint）；"
                  "测试集只能用于最终一次性评估，反复评测/据此调参视为泄露")
            if getattr(args, "allow_repeat_test", False) is not True:
                raise SystemExit(
                    f"拒绝重复 test 评测：checkpoint {ckpt_sha256[:16]} 已在 test "
                    "上评测过（账本见 data_dir/.test_access_log.jsonl）。测试集只"
                    "用于最终一次性评估；确需重测请显式加 --allow_repeat_test"
                    "（账本仍会记录，报告中须披露）。")
    # test 已在上面写入访问账本；此处显式放行是“尘封协议”的唯一合法出口
    val = load_bin(args.data_dir, split, allow_test=(split == "test"))
    # 与训练时的评估保持一致：优先使用训练配置里的 train_ctx
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config_used.json")
    train_ctx = None
    use_amp = amp_dtype_for(device, True) is not None
    cfg_used = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg_used = json.load(f)
        except (OSError, ValueError):
            cfg_used = None
        if not isinstance(cfg_used, dict):
            print(f"[eval] 警告：{cfg_path} 不是 JSON 对象，按缺失配置处理")
            cfg_used = {}
        train_ctx = cfg_used.get("train_ctx")
        # 与训练内评测口径一致：优先沿用训练时实际生效的精度（amp_dtype），
        # 避免在支持 bf16 的机器上把 fp32 回退训练的 checkpoint 用 bf16 评测。
        use_amp = _resolve_eval_amp(cfg_used, device)
    else:
        # 缺少训练配置时评测上下文静默变为 block_size，PPL 口径与训练内评测不一致
        print(f"[eval] 警告：未找到 {cfg_path}，评测上下文退化为 "
              f"model.block_size={model.block_size}，PPL 可能与训练内评测不可比")
    train_ctx = _positive_int_or_none(train_ctx)
    ctx = min(model.block_size, train_ctx) if train_ctx else model.block_size
    cfg_eval_iters = _positive_int_or_none(cfg_used.get("eval_iters"))
    # 缺省必须与 train.py 的 eval_iters 默认值（50）一致：否则同一 checkpoint
    # 的独立评测窗口数与训练内评测/选择口径不同，PPL 不可比，run_all 的窗口
    # 对齐检查也会因此回退到全 split 参考。
    eval_iters = (int(args.eval_iters) if args.eval_iters is not None
                  else (int(cfg_eval_iters) if cfg_eval_iters is not None
                        else 50))

    # 与训练内评测口径一致：尊重训练配置的 amp/eval_iters 设置（显式参数优先）。
    # evaluate_val 内部会把窗口不重复且均匀铺满整个验证集，避免循环复用样本。
    metrics = evaluate_val(model, val, ctx, device, eval_iters, amp=use_amp,
                           collect_window_losses=True)
    norm = json_safe_eval_metrics(metrics)
    ppl, vl, gmean = norm["val_ppl"], norm["val_loss"], norm["gate_mean"]
    if vl is None:
        print("[eval] 警告：验证损失非有限（NaN/Inf），该 checkpoint 已失效；"
              "val_ppl/val_loss 记为 null（避免写入非法 JSON 字面量、也避免 "
              "min(20, NaN) 恰为 exp(20) 被误读成截断 PPL），"
              "该结果不得与正常 PPL 比较")

    res = {
        "variant": ck.get("variant", model.variant),
        "seed": seed,
        "ckpt": args.ckpt,
        # 被评测产物的内容指纹：结果与 checkpoint 一一绑定，换/改 checkpoint
        # 后旧结果无法冒充（与 test 访问账本、实验台账中的哈希同一口径）
        "ckpt_sha256": ckpt_sha256,
        "split": split,
        # 评测前的数据协议结论随结果落盘，供汇总层区分正式/放行结果。
        "data_protocol_preflight": data_preflight,
        # checkpoint（model_best.pt）在 val 上选出；split=val 时报告指标与选择集同源
        "selection_set_is_eval_set": bool(split == "val"),
        "val_ppl": ppl,
        "val_loss": vl,
        # 通用别名：split="test" 时 val_ppl/val_loss 为兼容键名，消费方应优先用这里
        "ppl": ppl,
        "loss": vl,
        "val_ppl_clamped": norm["val_ppl_clamped"],
        "eval_ctx": ctx,
        # 评测实际使用的数值精度：旧产物无 amp_dtype 时也能从结果反查。
        "amp": bool(use_amp),
        "amp_dtype": (str(amp_dtype_for(device, use_amp)).replace("torch.", "")
                      if amp_dtype_for(device, use_amp) is not None else None),
        "gate_mean": gmean,
        "gate_std": metrics.get("gate_std"),
        "gate_frac_low": metrics.get("gate_frac_low"),
        "gate_frac_high": metrics.get("gate_frac_high"),
        "gate_layer_mean": metrics.get("gate_layer_mean"),
        "n_params": count_params(model),
        "n_gate_extra": gate_extra_params(model),
        # 开销披露：逐头参数量与模型结构一并落盘，summary 不必再从名字猜
        "n_gate_per_head": gate_extra_params_per_head(model),
        "n_heads": getattr(model, "n_heads", None),
        "n_layers": getattr(model, "n_layers", None),
        "d_head": (int(model.d_model // model.n_heads)
                   if getattr(model, "n_heads", 0) else None),
        **checkpoint_selection_info(
            ck, cfg_used.get("steps") if isinstance(cfg_used, dict) else None),
        "eval_iters_requested": metrics["eval_iters_requested"],
        "eval_iters": metrics["eval_iters"],
        "val_windows": metrics["val_windows"],
        "val_tokens_evaluated": metrics["val_tokens_evaluated"],
        # 逐窗口损失与窗口序号：同 split/同协议下各变体共享同一批窗口，
        # 下游可做窗口级配对（评估抽样不确定性），与跨种子方差分开报告。
        "window_losses": metrics.get("window_losses"),
        "window_indices": metrics.get("window_indices"),
        # 有限验证窗口的抽样误差：与 PPL 点估计同源，必须一起传递，否则
        # 下游只看到点估计，无法区分“真实差异”与“窗口抽样噪声”。
        "val_loss_se": metrics.get("val_loss_se"),
        "val_loss_ci95": metrics.get("val_loss_ci95"),
        "val_ppl_ci95": metrics.get("val_ppl_ci95"),
        "val_loss_min": metrics.get("val_loss_min"),
        "val_loss_max": metrics.get("val_loss_max"),
        # test 访问账本摘要（split=val 时为 None）：审计"测试集被反复评测"
        "test_access": test_access,
    }
    # 坏案例：把预测最差的窗口连同 top-5 预测一起落盘（PPL 只反映平均，
    # 最差窗口揭示模型盲区——生僻 token？不可预测噪声？特定模式崩溃？）
    # test 上转储默认关闭：最差窗口会连同上下文/预测暴露 test 内容，交人工
    # 迭代 = 据 test 改进模型（测试集变隐式训练集）。需要审阅时显式开启。
    worst_on_test_ok = worst_windows_allowed(
        split, getattr(args, "worst_on_test", False))
    if args.worst_n > 0 and not worst_on_test_ok:
        res["worst_windows_skipped_reason"] = "test_sealed"
        print("[eval] 已跳过 test 最差窗口转储（默认不在 test 上做坏案例迭代）；"
              "确需审阅 test 错误请显式加 --worst_on_test，并在报告中披露")
    if args.worst_n > 0 and worst_on_test_ok:
        res["worst_windows_on_test"] = bool(split == "test")
        try:
            tok = load_tokenizer(args.data_dir)
            res["worst_windows"] = collect_worst_windows(
                model, val, ctx, device, tok, n_worst=int(args.worst_n),
                amp=use_amp, split=split)
            # 把筛查范围随结果落盘：最差窗口来自“候选窗口子集”而非全量排序，
            # 汇总层必须能区分 64/665 与 64/64 两种覆盖度。
            res["worst_window_scope"] = worst_window_screening(
                len(val), ctx, WORST_SCREEN_MAX)
            print(f"[eval] 最差 {len(res['worst_windows'])} 个窗口已随结果落盘"
                  f"（候选 {res['worst_window_scope']['screened']}/"
                  f"{res['worst_window_scope']['total']} 个窗口；"
                  "res['worst_windows']）")
            if split == "test":
                # 落盘即意味着 test 内容进入可读报告：必须留痕，供审计。
                try:
                    test_access = append_test_access(args.data_dir, {
                        "time": now_str(),
                        "reason": "worst_window_dump_on_test",
                        "ckpt_sha256": ckpt_sha256,
                        "variant": ck.get("variant", model.variant),
                        "seed": seed,
                        "n_worst": int(args.worst_n),
                        "test_data_fingerprint": test_data_fingerprint,
                    })
                    # 结果 JSON 里的账本摘要必须包含刚写入的转储记录；
                    # 否则报告会少报一次 test 读取，审计链断开。
                    res["test_access"] = test_access
                except OSError as e:
                    print(f"[eval] 警告：test 最差窗口转储的访问记录写入失败"
                          f"（{e}）")
        except Exception as e:  # 坏案例转储失败不应阻塞评测本身
            print(f"[eval] 警告：最差窗口转储失败（{e}）")
    # 继承训练 run 的代码/数据溯源信息（run_manifest.json），保证结果可追溯
    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    prov = load_provenance(run_dir, args.data_dir)
    res.update(prov)
    # 与训练完成时写回 run_manifest 的 checkpoint 哈希交叉核对：目录里的
    # 权重若在训练后被替换/覆盖，eval.json 仍绑定实际内容，但必须显式标出
    # “它已不是训练清单记录的那一份”，不能让旧清单替新权重背书。
    expected_ckpt = ((prov.get("manifest_checkpoint_hashes") or {})
                     .get(os.path.basename(os.path.abspath(args.ckpt))))
    if isinstance(expected_ckpt, str):
        res["checkpoint_matches_manifest"] = (expected_ckpt == ckpt_sha256)
        if expected_ckpt != ckpt_sha256:
            print("[eval] ⚠ 被评测 checkpoint 的内容哈希与训练 run_manifest "
                  f"记录不一致（manifest={expected_ckpt[:16]}…, "
                  f"actual={ckpt_sha256[:16]}…）：权重可能已被替换/覆盖，"
                  "本次结果不能声称对应该清单的训练产物")
    else:
        res["checkpoint_matches_manifest"] = None
    if prov.get("data_fingerprint_matches_training") is False:
        print("[eval] ⚠ 当前 data_dir 的数据指纹与 checkpoint 训练清单不一致："
              "本次指标来自另一份数据，不能与训练时的 val/test 结果直接比较；"
              "已分别写入 eval_data_fingerprint / train_data_fingerprint")
    # 训练过程统计随被评测 checkpoint 一起绑定（run_all 从本文件读取）
    res["train_step_time_ms"], res["train_tokens_per_sec"] = \
        load_train_throughput(run_dir)
    res["train_peak_mem_mb"] = load_train_peak_mem_mb(run_dir)
    # 训练数据覆盖率随 checkpoint 绑定：epochs_seen<1 时下游必须把这份
    # 评测标为“未充分训练”，而不是与跑满数据的 run 直接比 PPL。
    res.update(load_train_coverage(run_dir))
    save_json(res, args.out)
    if ppl is None:
        print(f"[eval] {res['variant']}: val_ppl=null（验证损失非有限）-> {args.out}")
    else:
        print(f"[eval] {res['variant']}: val_ppl={ppl:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
