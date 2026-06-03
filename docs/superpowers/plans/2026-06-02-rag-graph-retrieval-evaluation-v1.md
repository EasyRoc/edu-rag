# Edu-RAG RAG Graph 与检索评估优化实施计划 V1

## Summary

第一版聚焦正确性与可评估性，保留现有 API、Milvus 数据和 RAGAS：

```text
classify
  ├─ non-educational -> chitchat -> finalize
  └─ educational -> retrieve -> rerank -> retrieval_gate
                                      ├─ accept  -> generate -> finalize
                                      ├─ retry   -> retry_planner -> retrieve
                                      └─ abstain -> abstain -> finalize
```

## Key Changes

- Graph 依赖通过闭包注入；`session_id -> user_id -> UUID4` 作为会话键。
- SSE 由服务层统一关闭队列，异常返回 `error` 和 `done`，不泄露内部错误。
- RRF 只排序并保留原始分数；在线门控只使用本地 CrossEncoder 的归一化 `rerank_score`。
- 第一次重试使用原问题和最多三个去重变体，第二次根据原因使用 HyDE 或 Step-Back。
- 低质量结果在重试耗尽后进入固定拒答节点，不调用答案生成器。
- 新增 `retrieval-evaluate` 和 `retrieval-calibrate` CLI，与 RAGAS 答案评估并列。

## Verification

- 单元测试覆盖门控、重排、RRF 字段、Graph 路由、SSE 生命周期、会话隔离、离线指标和校准。
- 保留原清洗、策略、app factory 和文档服务回归测试。
- 手工 smoke 覆盖 `/health`、`/docs`、普通问答、流式问答和拒答分支。

## Roadmap

后续阶段再实施 `AsyncPostgresSaver`、跨进程会话持久化、生成后事实校验、引用覆盖检查、并发检索优化、线上指标看板和灰区 LLM grader。
