# langgraph：状态图编排

langgraph 把智能体工作流建模为状态图（StateGraph）：节点是函数或 Runnable，边定义状态转移，一个共享 State 对象在图中流动，每个节点读写其中的字段。

与 AgentExecutor 的线性循环不同，状态图支持条件分支、循环与多智能体协作：多个智能体各占一个节点，通过共享状态接力，适合需要角色分工与审批的复杂任务。

人工审批是 langgraph 的招牌能力：编译图时设置 `interrupt_before`，执行到指定节点前会暂停，等人确认后以 `invoke(None, config)` 续跑；这对金融、运维等高风险操作场景是硬需求。

持久化由独立的 checkpoint 组件承担，当前版本 0.2.20，发布于 2025-02-18。它把每一步的 State 快照写入存储，既支撑 interrupt 之后的恢复执行，也让任意一次运行可以完整回放调试。

性能上，节点级并发由 `max_concurrency` 参数控制，防止单次运行打满外部服务配额；超时由 step_timeout 兜底，两者都可在编译参数中配置。langgraph 与 langchain-agents 共享工具定义与消息类型，从执行器迁移到状态图的主要工作量在把循环逻辑改写为图结构。
