# lega-attention 代码审查报告 v10

**审查时间**: 2026-11-25 23:10 CST  
**审查范围**: 全量 Python 代码（lega/ 目录下 50+ 模块，约 55,000 行）  
**审查方法**: AST 语法检查 + 静态分析 + 核心算法逐文件审读 + 统计函数运行验证

---

## 1. 总体结论

**未发现需要修复的 bug 或科研方法问题。** 该仓库的代码质量极高，经过多轮迭代审查，之前的 BUG FIX（analysis.py:1330 mask 长度覆盖问题）已被正确修复。统计方法设计严谨，覆盖了科研全流程。

## 2. 审查详情

### 2.1 语法与导入
- ✅ 全量 AST 解析无语法错误
- ✅ 所有核心模块（utils、attention、bayes、analysis 等）可正常导入
- ✅ VARIANTS 注册表完整（10 个主实验变体）

### 2.2 核心注意力机制 (lega/attention.py)
- ✅ LEGA 门控公式 `g = sigmoid(w·q + a·c + b*log n + d)` 实现正确
- ✅ 浓度统计量 `c = log(n * max_j p_j)` 的 clamp_min(1e-30) 处理了 log(0) 边界
- ✅ 位置置换函数 `_permute_positions` 正确处理了恒等变换回退（steps == 0 时用 T//2）
- ✅ 安慰剂证据 `_placebo_evidence` 使用确定性伪随机，不消耗全局 RNG

### 2.3 统计方法 (lega/utils.py)
- ✅ `bootstrap_mean_ci`: 百分位 bootstrap，处理了全相同值、NaN、n<2 等边界情况
- ✅ `paired_t_test`: 配对 t 检验，正确处理零方差（t = ±inf with correct sign）
- ✅ `paired_sign_flip_test`: n≤16 时全枚举精确 p 值，更大 n 用固定种子 Monte-Carlo
- ✅ `holm_bonferroni`: 逐步下降法校正，保证校正后 p 单调不减
- ✅ `wilson_score_ci`: Wilson 得分区间，小样本下远优于 Wald 区间
- ✅ `min_detectable_effect`: MDE 公式正确，处理了极大 n 的 OverflowError

### 2.4 效应量与贝叶斯分析
- ✅ `_compute_cohens_d` 含 Hedges' g 校正：`correction = 1 - 3/(4*(n1+n2) - 9)`
- ✅ 贝叶斯因子 BF10 的解读分级（极强/强/中等/弱/不足）符合 Jeffreys 标准
- ✅ 多处 Cohen's d 实现一致（bayes.py, curves.py, discovery.py 等 7 个模块）

### 2.5 科研方法
- ✅ 多重比较校正：Holm-Bonferroni 已实现且被调用
- ✅ 效应量报告：Cohen's d + Hedges' g 在所有对比中均有
- ✅ 预注册与假设检验：贝叶斯证据层（H0/H1 判定）完备
- ✅ 数据泄露防护：消融实验在 val split 上执行，test 已尘封
- ✅ 可重复性：PYTHONHASHSEED、deterministic algorithms、线程固定均有保障

### 2.6 异常处理
- ✅ `except Exception` 均有明确注释说明（如 "pragma: no cover - 只在极端环境"）
- ✅ 无裸 `except:` 或 `pass` 吞掉异常的情况

## 3. 运行验证

| 测试 | 结果 |
|------|------|
| `bootstrap_mean_ci([5,5,5,5])` | (5.0, 5.0) ✅ |
| `bootstrap_mean_ci([1.0, nan, 3.0, 4.0])` | (1.0, 4.0) ✅ |
| `holm_bonferroni([0.01, 0.04, 0.03, 0.20])` | [0.04, 0.09, 0.09, 0.20] ✅ |
| `paired_t_test` 零方差 | t=-inf, p=0.0 ✅ |

## 4. 代码质量评级

| 维度 | 评级 |
|------|------|
| 语法正确性 | A |
| 数值稳定性 | A |
| 统计方法严谨性 | A+ |
| 科研方法规范性 | A+ |
| 可重复性 | A+ |
| 异常处理 | A |
| 文档完整性 | A |
