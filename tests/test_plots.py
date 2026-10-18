"""可视化结果文件选择测试：同一 (变体, 种子) 不能重复计数。

在旧结果目录上以不同种子数重跑时，裸命名文件与 _seed<s> 文件可能同时
存在（并指向同一个种子）。图表的均值/误差棒必须按 repro.json 的命名
约定只取本次 run 的产物，否则同一次实验会被多算一次、过期结果混入均值。
"""
import json
import os
import shutil
import tempfile

from lega.plots import (_expected_tagged, _group_by_variant, _prefer_seeded)


def _w(rdir, name, obj):
    with open(os.path.join(rdir, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def test_group_by_variant_deduplicates_seed_slots_by_convention():
    tmp = tempfile.mkdtemp()
    try:
        # 多种子约定：过期的裸命名文件不得与 _seed1 重复计入
        _w(tmp, "repro.json", {"seeds": [1, 2]})
        _w(tmp, "eval_lega.json", {"variant": "lega", "seed": 1, "val_ppl": 999.0})
        _w(tmp, "eval_lega_seed1.json", {"variant": "lega", "seed": 1, "val_ppl": 10.0})
        _w(tmp, "eval_lega_seed2.json", {"variant": "lega", "seed": 2, "val_ppl": 11.0})
        g = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert sorted(d["val_ppl"] for d in g["lega"]) == [10.0, 11.0]
        # 单种子约定：当前裸命名文件胜出，过期的 _seed 文件不得混入
        _w(tmp, "repro.json", {"seeds": [1]})
        _w(tmp, "eval_lega.json", {"variant": "lega", "seed": 1, "val_ppl": 12.0})
        g2 = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert sorted(d["val_ppl"] for d in g2["lega"]) == [12.0]
        # 无 repro 信息：退回旧行为（_seed 优先，与 _seed1 冲突的裸文件丢弃）
        os.remove(os.path.join(tmp, "repro.json"))
        g3 = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert sorted(d["val_ppl"] for d in g3["lega"]) == [10.0, 11.0]
        # 无任何种子信息才保留（旧格式单文件）
        for name in ("eval_lega_seed1.json", "eval_lega_seed2.json"):
            os.remove(os.path.join(tmp, name))
        _w(tmp, "eval_lega.json", {"variant": "lega", "val_ppl": 8.0})
        g4 = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert [d["val_ppl"] for d in g4["lega"]] == [8.0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 变体分组：按 (变体, 种子) 去重且随命名约定择优")


def test_prefer_seeded_matches_run_convention():
    tmp = tempfile.mkdtemp()
    try:
        bare = os.path.join(tmp, "analysis_records_lega.csv")
        tagged = os.path.join(tmp, "analysis_records_lega_seed1.csv")
        for p in (bare, tagged):
            open(p, "w").close()
        _w(tmp, "repro.json", {"seeds": [1, 2]})
        assert _expected_tagged(tmp) is True
        assert _prefer_seeded([bare, tagged], rdir=tmp) == [tagged]
        _w(tmp, "repro.json", {"seeds": [1]})
        assert _expected_tagged(tmp) is False
        assert _prefer_seeded([bare, tagged], rdir=tmp) == [bare]
        # 无 repro：_seed 优先（旧行为）
        os.remove(os.path.join(tmp, "repro.json"))
        assert _expected_tagged(tmp) is None
        assert _prefer_seeded([bare, tagged], rdir=tmp) == [tagged]
        # 约定族不存在时回退到另一族，而不是画不出图
        os.remove(bare)
        _w(tmp, "repro.json", {"seeds": [1]})
        assert _prefer_seeded([bare, tagged], rdir=tmp) == [tagged]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 记录文件选择：按 repro.json 命名约定取本次产物")
