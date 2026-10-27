"""从 test_run_all.py 拆出的共享测试夹具。"""
import json
import os


def _mk(cand, ref, metric, mean_diff, p, n=5, lower=True):
    return {"candidate": cand, "reference": ref, "metric": metric,
            "mean_diff": mean_diff, "std_diff": 0.1, "p": p, "p_adj": None,
            "n": n, "lower_is_better": lower, "d_z": 1.0}


def _write_json(path, obj):
    # 测试 fixture 默认模拟“配置显式统一走显式注意力路径”；需要测试缺键/
    # 默认值行为的用例直接写文件，避免被这里的默认值掩盖。
    if (os.path.basename(path) == "config_used.json"
            and isinstance(obj, dict) and "use_sdpa_standard" not in obj):
        obj = dict(obj, use_sdpa_standard=False)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
