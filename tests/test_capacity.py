"""训练期容量匹配分解的纯计算测试。"""
from lega.capacity import (CAPACITY_NULL_VARIANTS, CAPACITY_OFF_VARIANTS,
                           NULL_EVIDENCE_COMPONENT, component_effects)


def test_component_mapping_is_bijective():
    assert set(CAPACITY_NULL_VARIANTS) == {"A", "B", "C"}
    assert set(CAPACITY_OFF_VARIANTS) == {"A", "B", "C"}
    assert NULL_EVIDENCE_COMPONENT == {
        v: k for k, v in CAPACITY_NULL_VARIANTS.items()}


def test_ppl_decomposition_identity_and_direction():
    # PPL 越低越好：Full=10、Null=12、No=15 -> 信息 2、容量 3、总 5。
    out = component_effects([10.0, 11.0], [12.0, 13.0], [15.0, 16.0],
                            lower_better=True)
    assert out["n"] == 2
    assert out["effects"]["info"]["values"] == [2.0, 2.0]
    assert out["effects"]["capacity"]["values"] == [3.0, 3.0]
    assert out["effects"]["total"]["values"] == [5.0, 5.0]
    assert out["effects"]["total"]["mean"] == 5.0
    assert out["info_share"] == 0.4
    assert out["identity_residual_max_abs"] == 0.0
    print("  [ok] PPL：info + capacity = total，正值表示 Full 更好")


def test_higher_better_decomposition_identity():
    # RULER EM 越高越好：Full=70、Null=60、No=50 -> 信息 10、容量 10、总 20。
    out = component_effects([70.0], [60.0], [50.0], lower_better=False)
    assert out["effects"]["info"]["values"] == [10.0]
    assert out["effects"]["capacity"]["values"] == [10.0]
    assert out["effects"]["total"]["values"] == [20.0]
    assert out["identity_residual_max_abs"] == 0.0
    print("  [ok] RULER：同一方向约定，正值表示 Full 更好")


def test_component_effects_rejects_misaligned_or_bad_values():
    for args in (([1.0, 2.0], [1.0], [1.0]),
                 ([1.0], [1.0], [float("nan")]),
                 ([True], [1.0], [1.0]),
                 ([1.0], [1.0], ["x"])):
        try:
            component_effects(*args, lower_better=True)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝非法输入: {args!r}")
    assert component_effects([1.0], [1.0], [1.0],
                             lower_better=True)["info_share"] is None
    print("  [ok] 拒绝未对齐/非有限/布尔/非数值输入")
