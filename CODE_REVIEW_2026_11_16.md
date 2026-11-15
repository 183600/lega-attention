# Code Review Report — lega-attention
**日期**: 2026-11-16
**审查范围**: lega/ 目录（44,012行Python代码）+ tests/ 目录

## 审查摘要

对 lega-attention 仓库进行了全面的代码审查。该仓库实现了 LEGA（Length-Explicit Gated Attention）机制的完整研究管线，包括训练、评估、消融实验和统计分析。

## 发现的问题

### 误报分析（自动扫描标记但确认为安全的项）

#### 1. eval() 调用 — 全部为误报
- 自动扫描标记了10处 `eval()` 调用为高风险代码注入风险
- 经 AST（抽象语法树）分析确认：所有调用均为 `model.eval()`（PyTorch 模型评估模式切换）
- **结论**: 无安全风险，不存在 Python 内置 `eval()` 的使用

#### 2. 除零保护 — 已有完善防护
- `lega/sanity.py:522` — 该行位于注释/文档中，实际代码已在上方验证 `log_every > 0`
- `lega/attention.py:930` — `d_model // n_heads`，代码注释明确说明"d_model % n_heads == 0 validated above"
- `lega/ablate.py:1311` — `/counts[li]`，第1308行已有 `counts[li] <= 0` 检查
- `lega/ablate.py:1612` — `std / math.sqrt(n)`，第1611行已有 `paired.get("n", 0) >= 2` 检查
- **结论**: 所有标记的除法均有保护，无需修复

### 审查通过的项目

#### 科研方法论
- ✅ **数据分割协议**: 严格的 seed-owned stream 分割，VAL_SEED/TEST_SEED/STATS_SEED 分离，训练入口拒绝使用评估种子
- ✅ **时间序列分割**: 提供 `time_ordered_split()` 和 `group_split()` 原语，包含完整性验证
- ✅ **评估梯度管理**: `evaluate_val()` 使用 `model.eval()` + `with torch.inference_mode()`
- ✅ **消融实验设计**: 完整的消融对照设计（Full / 证据置零 / 参数+证据都去掉），使 `total = info + capacity` 精确成立
- ✅ **多重比较校正**: `exploratory_fdr()` 实现了 BH 和 BY 两种 FDR 控制
- ✅ **数值稳定性**: 门控统计使用 float64 累加，防止 fp32 抵消；loss 累加器使用 float64

#### 代码质量
- **防御性编程**: `strict_int()` 输入验证、非有限值检测、窗口边界检查
- **文档质量**: 详尽的中文注释，包含数学公式推导和设计决策解释
- **测试覆盖**: 全面的单元测试套件，覆盖边界条件和数值精度

## 代码质量评估

- **整体质量**: 优秀
- **主要优势**: 防御性编程全面、方法论严谨、文档详尽
- **无需要修复的问题**: 所有自动扫描标记的问题均为误报或已有防护
