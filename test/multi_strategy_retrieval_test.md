# 多策略混合检索 测试文档

## 1. 功能概述

在现有混合检索基础上引入5种查询策略，根据意图、复杂度和检索质量自动选择。详见 [spec/multi_strategy_retrieval.md](../spec/multi_strategy_retrieval.md)。

## 2. 测试策略

### 2.1 单元测试（白盒）

- **策略选择器**：覆盖所有 intent × complexity 组合的 `select_strategy` 逻辑
- **质量评估**：`assess_retrieval_quality` / `should_apply_hyde` / `should_apply_step_back` 的阈值边界
- **多查询融合**：`multi_query_fusion` RRF 排序正确性、去重、分数归一化、不修改原始数据
- **子结果合并**：`merge_sub_results` 去重保留最高分逻辑
- **LLM 响应解析**：变体生成/问题拆解的输出格式解析（编号去除、空行过滤）

### 2.2 集成测试（灰盒）

- **DIRECT 策略端到端**：完整 `hybrid_retrieve` 调用（simple 查询），验证不触发额外 LLM 调用
- **策略降级**：LLM 不可用时 MULTI_QUERY 和 DECOMPOSITION 降级为 DIRECT，不报错
- **补充策略流程**：低质量检索结果的 Step-Back/HyDE 补充逻辑（mock LLM）
- **retrieve_node 集成**：策略选择 → 检索 → 评估的完整链路

## 3. 测试用例

| 编号 | 场景 | 输入 | 期望输出 | 覆盖的边界条件 |
|------|------|------|----------|----------------|
| U01 | simple 查询选 DIRECT | intent=educational, complexity=simple | StrategyType.DIRECT | 基态 |
| U02 | medium 查询选 MULTI_QUERY | intent=educational, complexity=medium | StrategyType.MULTI_QUERY | 基态 |
| U03 | complex 查询选 DECOMPOSITION | intent=educational, complexity=complex | StrategyType.DECOMPOSITION | 基态 |
| U04 | 非教育意图一律 DIRECT | intent=chitchat/complexity=complex | StrategyType.DIRECT | 意图分流 |
| U05 | 高质量结果通过评估 | 3+ docs, avg≥0.5 | assess=True | 阈值边界 |
| U06 | 低平均分不通过 | avg<0.5 | assess=False | 阈值边界 |
| U07 | 结果不足不通过 | <3 docs | assess=False | 数量边界 |
| U08 | 空结果不通过 | [] | assess=False, hyde=True, step_back=True | 空集处理 |
| U09 | top1<0.4 触发 HyDE | [score=0.3, ...] | hyde=True | HyDE 阈值 |
| U10 | top1≥0.4 不触发 HyDE | [score=0.9, ...] | hyde=False | HyDE 阈值 |
| U11 | 多查询融合排序正确 | 3路含重复chunk | 按RRF降序，去重 | RRF k=60 |
| U12 | 融合不修改原始数据 | 原始score=0.9 | 融合后原始dict score不变 | 副作用 |
| U13 | 合并保留最高分 | 同一chunk在两路中score=0.7/0.8 | 保留0.8 | 去重保留最高 |
| U14 | 变体解析过滤编号 | "1. xxx\n2. yyy" | ["xxx", "yyy"] | 格式清理 |
| U15 | 空LLM返回→降级 | llm返回"" | 返回空列表/[原始query] | LLM故障 |
| I01 | DIRECT端到端 | simple查询 + 有数据Milvus | 返回结果，无LLM调用 | 端到端集成 |
| I02 | LLM不可用时降级 | 无API_KEY + medium查询 | 降级为DIRECT，正常返回结果 | 降级容错 |
| I03 | retrieve_node策略标记 | educational+medium查询 | 走MULTI_QUERY路径 | 节点集成 |
| I04 | re_retrieve用直接检索 | 任意查询 | 直接调hybrid_search(top_k=8) | 重试不走策略 |

## 4. 测试环境 & 数据

- **依赖**：无 LLM API Key 也可运行（单元测试不调用 LLM，集成测试降级验证）
- **Milvus**：集成测试需要已初始化的 `milvus_k12.db`（含测试数据）
- **Python**：3.10+，无额外 pip 依赖（不使用 pytest/mock）
- **数据**：单元测试全部使用构造数据；集成测试依赖现有 Milvus 数据

## 5. 运行方式

```bash
# 全部测试
python test/test_strategies.py

# 仅单元测试（无需 Milvus/LLM，始终可运行）
python test/test_strategies.py --unit-only

# 仅集成测试（需要 Milvus 数据）
python test/test_strategies.py --integration-only

# 详细输出
python test/test_strategies.py --verbose
```
