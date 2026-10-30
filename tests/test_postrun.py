"""评测流水线链（lega.postrun）测试：阶段命令逐字段等价、退出码传播、argv 还原。

postrun 把同一 run 的 train/evaluate/ruler/analysis 放进一个进程顺序执行，
是 run_all 流水线的主干。这里不拉起真实训练（真实训练由冒烟流水线覆盖），
只覆盖三类易错点：
  * 阶段命令构造必须与 run_all 的独立命令逐字段一致（参数名/顺序/冒烟
    透传），否则评测口径会静默漂移；
  * runpy 执行子模块时 sys.argv 必须被替换并还原，SystemExit(0/None) 视为
    成功、非零码原样传播（run_all 依赖退出码定位失败阶段）；
  * checkpoint 解析口径：model_best.pt 优先、缺失回退 model.pt、都无则跳过。
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

from lega.postrun import (_analysis_argv, _eval_argv, _parser, _ruler_argv,
                          _run_module, _train_argv, resolve_ckpt)

STUB = '''import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(sys.argv, f)
code = int(sys.argv[2])
if code:
    raise SystemExit(code)
'''


def _make_stub(dirpath: str) -> str:
    with open(os.path.join(dirpath, "postrun_stub_cli.py"), "w",
              encoding="utf-8") as f:
        f.write(STUB)
    return dirpath


def test_run_module_argv_replaced_and_restored():
    tmp = tempfile.mkdtemp(prefix="postrun_stub_")
    out = os.path.join(tmp, "argv.json")
    try:
        _make_stub(tmp)
        sys.path.insert(0, tmp)
        old_argv = list(sys.argv)
        _run_module("postrun_stub_cli", [out, "0", "--flag"])
        assert sys.argv == old_argv, "sys.argv 必须还原"
        with open(out, encoding="utf-8") as f:
            seen = json.load(f)
        assert seen == ["-m postrun_stub_cli", out, "0", "--flag"], seen
    finally:
        if tmp in sys.path:
            sys.path.remove(tmp)
        sys.modules.pop("postrun_stub_cli", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] _run_module 替换/还原 sys.argv，正常返回视作成功")


def test_run_module_exit_code_propagation():
    tmp = tempfile.mkdtemp(prefix="postrun_stub_")
    try:
        _make_stub(tmp)
        sys.path.insert(0, tmp)
        old_argv = list(sys.argv)
        _run_module("postrun_stub_cli", [os.path.join(tmp, "a.json"), "0"])
        try:
            _run_module("postrun_stub_cli", [os.path.join(tmp, "b.json"), "7"])
            raise AssertionError("SystemExit(7) 必须原样传播")
        except SystemExit as e:
            assert e.code == 7, e.code
        assert sys.argv == old_argv
    finally:
        if tmp in sys.path:
            sys.path.remove(tmp)
        sys.modules.pop("postrun_stub_cli", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] _run_module 传播非零退出码")


def _ns():
    return _parser().parse_args([
        "--config", "configs/tiny.json", "--variant", "lega",
        "--seed", "1337", "--data_dir", "data/dummy",
        "--out_dir", "out/tiny_smoke/lega",
        "--results_dir", "results/tiny_smoke", "--suffix", "",
    ])


def test_stage_argv_matches_cli():
    a = _ns()
    ckpt = "out/tiny_smoke/lega/model_best.pt"
    assert _train_argv(a) == [
        "--config", "configs/tiny.json", "--variant", "lega",
        "--data_dir", "data/dummy", "--out_dir", "out/tiny_smoke/lega",
        "--seed", "1337", "--device", "auto"]
    assert _eval_argv(a, ckpt) == [
        "--ckpt", ckpt, "--data_dir", "data/dummy",
        "--out", os.path.join("results/tiny_smoke", "eval_lega.json"),
        "--worst_n", "10", "--device", "auto"]
    # 显式放行 test 最差窗口时，标志位必须透传到 evaluate
    a_test = _parser().parse_args([
        "--config", "configs/tiny.json", "--variant", "lega",
        "--seed", "1337", "--data_dir", "data/dummy",
        "--out_dir", "out/tiny_smoke/lega",
        "--results_dir", "results/tiny_smoke", "--suffix", "",
        "--worst_on_test", "--allow_repeat_test"])
    assert "--worst_on_test" in _eval_argv(a_test, ckpt)
    assert "--allow_repeat_test" in _eval_argv(a_test, ckpt)
    assert "--worst_on_test" not in _eval_argv(a, ckpt)
    assert "--allow_repeat_test" not in _eval_argv(a, ckpt)
    assert _ruler_argv(a, ckpt) == [
        "--ckpt", ckpt, "--data_dir", "data/dummy",
        "--out", os.path.join("results/tiny_smoke", "ruler_lega.json"),
        "--device", "auto"]
    assert _analysis_argv(a, ckpt) == [
        "--ckpt", ckpt, "--data_dir", "data/dummy",
        "--out_dir", "results/tiny_smoke", "--device", "auto", "--suffix", ""]
    print("  [ok] 阶段命令与 run_all 独立命令逐字段一致")


def test_smoke_and_seed_suffix_transparency():
    a = _parser().parse_args([
        "--config", "configs/tiny.json", "--variant", "lega_no_len",
        "--seed", "2", "--data_dir", "data/dummy",
        "--out_dir", "out/tiny_smoke/lega_no_len_seed2",
        "--results_dir", "results/tiny_smoke", "--suffix", "_seed2",
        "--smoke", "--wandb",
        "--early_stop_patience", "5",
        "--ruler_ctx_lens", "256", "--ruler_n_samples", "2",
        "--analysis_n_examples", "2", "--analysis_ctx", "256",
        "--skip_analysis",
    ])
    ckpt = "out/tiny_smoke/lega_no_len_seed2/model.pt"
    t = _train_argv(a)
    assert t[t.index("--smoke"):] == ["--smoke", "--wandb",
                                      "--early_stop_patience", "5"], t
    r = _ruler_argv(a, ckpt)
    assert r[-4:] == ["--ctx_lens", "256", "--n_samples", "2"], r
    assert r[r.index("--out") + 1] == os.path.join(
        "results/tiny_smoke", "ruler_lega_no_len_seed2.json")
    an = _analysis_argv(a, ckpt)
    assert an[-4:] == ["--n_examples", "2", "--ctx", "256"], an
    assert an[an.index("--suffix") + 1] == "_seed2"
    assert a.skip_analysis and not a.skip_ruler and not a.skip_train
    print("  [ok] 冒烟透传/种子后缀/skip 开关映射正确")


def test_resolve_ckpt_prefers_best():
    tmp = tempfile.mkdtemp(prefix="postrun_ckpt_")
    try:
        assert resolve_ckpt(tmp) is None
        plain = os.path.join(tmp, "model.pt")
        open(plain, "w").close()
        assert resolve_ckpt(tmp) == plain
        best = os.path.join(tmp, "model_best.pt")
        open(best, "w").close()
        assert resolve_ckpt(tmp) == best
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] resolve_ckpt: best 优先 / 回退 model.pt / 都无返回 None")


def test_parser_defaults():
    a = _ns()
    assert a.device == "auto" and a.worst_n == 10 and a.suffix == ""
    assert not (a.smoke or a.wandb or a.skip_train or a.skip_eval
                or a.skip_ruler or a.skip_analysis)
    assert a.ruler_ctx_lens is None and a.early_stop_patience is None
    assert a.sanity_file is None
    assert isinstance(a, argparse.Namespace)
    print("  [ok] CLI 默认值与 skip 开关缺省关闭")


def test_train_argv_forwards_sanity_file_only_when_present():
    """sanity 证据透传：提供时进入 train 命令，缺省不夹带旧文件。"""
    a = _ns()
    assert "--sanity_file" not in _train_argv(a)
    a2 = _parser().parse_args([
        "--config", "configs/tiny.json", "--variant", "lega",
        "--seed", "1337", "--data_dir", "data/dummy",
        "--out_dir", "out/tiny_smoke/lega",
        "--results_dir", "results/tiny_smoke", "--suffix", "",
        "--sanity_file", "results/tiny_smoke/sanity_lega.json",
    ])
    t = _train_argv(a2)
    assert t[t.index("--sanity_file") + 1] == \
        "results/tiny_smoke/sanity_lega.json"
    print("  [ok] postrun 透传 --sanity_file（缺省不夹带）")


def test_full_eval_flag_is_forwarded_to_evaluate():
    """全量窗口评测开关必须从 postrun 透传到 evaluate，不能被静默丢弃。"""
    a = _parser().parse_args([
        "--config", "configs/tiny.json", "--variant", "lega",
        "--seed", "1337", "--data_dir", "data/dummy",
        "--out_dir", "out/tiny_smoke/lega",
        "--results_dir", "results/tiny_smoke", "--full_eval",
    ])
    argv = _eval_argv(a, "out/tiny_smoke/lega/model_best.pt")
    assert "--full_eval" in argv
    # 缺省仍沿用配置里的抽样窗口，避免无意中改变既有口径。
    assert "--full_eval" not in _eval_argv(_ns(), "ckpt")
    print("  [ok] --full_eval 仅在显式开启时透传 evaluate")
