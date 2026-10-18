"""单 run 流水线链：训练 → 独立评测 → RULER → 机制分析（同一进程内顺序执行）。

每个阶段仍以自己的 main() 运行（runpy 以 ``__main__`` 语义执行
``lega.<stage>``，sys.argv 临时替换为等价命令），各阶段 main() 自行
set_seed / 加载模型 / 写自己的输出文件，阶段之间没有共享状态；
PYTHONHASHSEED 由外层命令的 --seed 确定（与 run_all._cmd_env 一致）。
文件路径、JSON 内容、随机源与独立进程逐条执行完全一致。

用法（由 run_all 逐 run 调用，也可手动排障）:
  python -m lega.postrun --config configs/tiny.json --variant lega --seed 1337 \\
      --data_dir data/dummy --out_dir out/tiny_smoke/lega \\
      --results_dir results/tiny_smoke --suffix "" --device auto [--smoke]
退出码：0=全部执行成功；非零=失败阶段的退出码（stderr 标明阶段名与等价命令）。
"""
import argparse
import os
import runpy
import sys
from typing import List, Optional

GATED_FOR_ANALYSIS_HINT = (
    "机制分析只对门控变体有意义；调用方（run_all）用 --skip_analysis 表达")


def resolve_ckpt(out_dir: str) -> Optional[str]:
    """评测用 checkpoint：优先验证集选出的最优权重，缺失时回退最终步模型。

    与 run_all 口径一致（model selection 用 model_best.pt；--skip_train
    复用旧产物时可能只有 model.pt）；都不存在返回 None（跳过评测阶段）。
    """
    best = os.path.join(out_dir, "model_best.pt")
    if os.path.exists(best):
        return best
    plain = os.path.join(out_dir, "model.pt")
    return plain if os.path.exists(plain) else None


def _train_argv(a) -> List[str]:
    argv = ["--config", a.config, "--variant", a.variant,
            "--data_dir", a.data_dir, "--out_dir", a.out_dir,
            "--seed", str(a.seed), "--device", a.device]
    if a.smoke:
        argv.append("--smoke")
    if a.wandb:
        argv.append("--wandb")
    if a.early_stop_patience is not None:
        argv += ["--early_stop_patience", str(a.early_stop_patience)]
    return argv


def _eval_argv(a, ckpt: str) -> List[str]:
    return ["--ckpt", ckpt, "--data_dir", a.data_dir,
            "--out", os.path.join(a.results_dir,
                                  f"eval_{a.variant}{a.suffix}.json"),
            "--worst_n", str(a.worst_n), "--device", a.device]


def _ruler_argv(a, ckpt: str) -> List[str]:
    argv = ["--ckpt", ckpt, "--data_dir", a.data_dir,
            "--out", os.path.join(a.results_dir,
                                  f"ruler_{a.variant}{a.suffix}.json"),
            "--device", a.device]
    if a.ruler_ctx_lens:
        argv += ["--ctx_lens", str(a.ruler_ctx_lens)]
    if a.ruler_n_samples is not None:
        argv += ["--n_samples", str(a.ruler_n_samples)]
    return argv


def _analysis_argv(a, ckpt: str) -> List[str]:
    argv = ["--ckpt", ckpt, "--data_dir", a.data_dir,
            "--out_dir", a.results_dir, "--device", a.device,
            "--suffix", a.suffix]
    if a.analysis_n_examples is not None:
        argv += ["--n_examples", str(a.analysis_n_examples)]
    if a.analysis_ctx is not None:
        argv += ["--ctx", str(a.analysis_ctx)]
    return argv


def _run_module(mod_path: str, argv) -> None:
    """在进程内以 ``python -m <mod_path>`` 的语义执行 CLI。

    仅临时替换 sys.argv（argparse 读取处）；runpy 默认不修改 sys.modules，
    被执行的模块不会残留为已导入状态。子模块正常返回或 SystemExit(0/None)
    视作成功；SystemExit(非零) 原样抛出，由 _run_stage 归类为阶段失败。
    """
    old_argv = sys.argv
    sys.argv = [f"-m {mod_path}"] + [str(x) for x in argv]
    try:
        runpy.run_module(mod_path, run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = old_argv


def _run_stage(name: str, mod_path: str, argv) -> None:
    print(f"[postrun] 阶段 {name}: python -m {mod_path} "
          f"{' '.join(str(x) for x in argv)}", flush=True)
    try:
        _run_module(mod_path, argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if code == 0:
            return
        print(f"[postrun] 阶段 {name} 失败（退出码 {code}）: python -m {mod_path} "
              f"{' '.join(str(x) for x in argv)}", flush=True)
        raise SystemExit(code)
    except BaseException:
        print(f"[postrun] 阶段 {name} 异常终止: python -m {mod_path} "
              f"{' '.join(str(x) for x in argv)}", flush=True)
        raise


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="单 run 流水线链：train → evaluate → ruler → analysis"
                    "（同一进程内顺序执行，产物与独立进程完全一致）")
    p.add_argument("--config", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True, help="训练输出目录（checkpoint 所在）")
    p.add_argument("--results_dir", required=True, help="评测/分析结果目录")
    p.add_argument("--suffix", default="",
                   help="多种子运行的文件名后缀（如 _seed1）")
    p.add_argument("--device", default="auto")
    p.add_argument("--worst_n", type=int, default=3,
                   help="evaluate 记录的最差窗口数")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--ruler_ctx_lens", default=None,
                   help="透传 ruler 的 --ctx_lens（逗号分隔）")
    p.add_argument("--ruler_n_samples", type=int, default=None)
    p.add_argument("--analysis_n_examples", type=int, default=None)
    p.add_argument("--analysis_ctx", type=int, default=None)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--skip_ruler", action="store_true")
    p.add_argument("--skip_analysis", action="store_true",
                   help=GATED_FOR_ANALYSIS_HINT)
    return p


def main() -> None:
    args = _parser().parse_args()

    if not args.skip_train:
        _run_stage("train", "lega.train", _train_argv(args))
    ckpt = resolve_ckpt(args.out_dir)
    if ckpt is None:
        print(f"[postrun] 跳过 {args.variant}/seed{args.seed}：无 checkpoint"
              f"（{args.out_dir}）", flush=True)
        return
    if not args.skip_eval:
        _run_stage("evaluate", "lega.evaluate", _eval_argv(args, ckpt))
    if not args.skip_ruler:
        _run_stage("ruler", "lega.ruler", _ruler_argv(args, ckpt))
    if not args.skip_analysis:
        _run_stage("analysis", "lega.analysis", _analysis_argv(args, ckpt))


if __name__ == "__main__":
    main()
