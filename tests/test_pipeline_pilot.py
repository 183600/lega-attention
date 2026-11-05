"""试点模式与评测 split 的流水线级测试。

覆盖：
  * postrun 支持 --eval_split，并把 split 透传给独立评测阶段；
  * run_all 的 --pilot 强制只评 val：与 --worst_on_test/--allow_repeat_test
    冲突时直接拒绝（试点不得访问 test）；
  * 试点模式在设计计划里留痕（run_mode 检查 + inputs.pilot）。
"""

import os
import sys
import tempfile

from lega.postrun import _eval_argv
from lega.pipeline.train import main as pipeline_main


class _Args:
    variant = "lega"
    suffix = "_seed1"
    results_dir = "results/x"
    data_dir = "data/x"
    device = "cpu"
    worst_n = 0
    full_eval = False
    window_offsets = None
    position_ppl = False
    worst_on_test = False
    allow_repeat_test = False
    eval_split = "val"


def _run_pipeline(argv):
    """以给定参数调用流水线入口，返回退出码（argparse 错误为 2）。"""
    old = sys.argv
    sys.argv = ["run_all.py"] + list(argv)
    try:
        pipeline_main()
    except SystemExit as exc:
        return exc.code
    finally:
        sys.argv = old
    return 0


def test_postrun_eval_argv_carries_split():
    """--eval_split 必须透传给 evaluate 的 --split。"""
    argv = _eval_argv(_Args(), "out/model.pt")
    assert "--split" in argv
    assert argv[argv.index("--split") + 1] == "val"


def test_postrun_parser_accepts_eval_split():
    """postrun 的 CLI 接受 --eval_split val。"""
    from lega.postrun import _parser

    args = _parser().parse_args(
        [
            "--config",
            "configs/tiny.json",
            "--variant",
            "lega",
            "--seed",
            "1337",
            "--data_dir",
            "data/dummy",
            "--out_dir",
            "out/tiny/lega",
            "--results_dir",
            "results/tiny",
            "--eval_split",
            "val",
        ]
    )
    assert args.eval_split == "val"


def test_run_all_pilot_rejects_test_access_flags():
    """--pilot 与 --worst_on_test/--allow_repeat_test 冲突时必须拒绝。"""
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = os.path.join(tmp, "results")
        code = _run_pipeline(
            [
                "--config",
                "configs/tiny.json",
                "--data_dir",
                "data/dummy",
                "--results_dir",
                results_dir,
                "--pilot",
                "--worst_on_test",
            ]
        )
        assert code == 2


def test_run_all_pilot_rejects_test_eval_split():
    """--pilot 不允许 --eval_split test。"""
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = os.path.join(tmp, "results")
        code = _run_pipeline(
            [
                "--config",
                "configs/tiny.json",
                "--data_dir",
                "data/dummy",
                "--results_dir",
                results_dir,
                "--pilot",
                "--eval_split",
                "test",
            ]
        )
        assert code == 2


def test_design_plan_marks_pilot_mode():
    """试点模式在设计计划里显式留痕。"""
    from lega.design import build_design_plan

    plan = build_design_plan(
        {"seeds": [900001, 900002, 900003], "steps": 10, "design": {}},
        [900001, 900002, 900003],
        ["lega"],
        pilot=True,
    )
    checks = {c["id"]: c for c in plan["checks"]}
    assert checks["run_mode"]["status"] == "info"
    assert "试点" in checks["run_mode"]["detail"]
    assert plan["inputs"]["pilot"] is True
