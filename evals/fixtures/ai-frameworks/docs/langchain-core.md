# langchain-core：核心架构与运行时

langchain-core 是 LangChain 生态的最底层核心库，提供 Runnable 抽象、LCEL（LangChain Expression Language）管道与基础回调设施。所有上层组件（agents、memory、retrievers）都构建在它之上。

LCEL 用竖线 `|` 把组件串成管道，例如 `prompt | model | parser`，运行时按顺序传递数据。每个 Runnable 都自动获得 `invoke`、`stream`、`batch` 三种调用方式，无需为流式或并发场景单独写适配代码。

容错方面，核心库提供 RunnableRetry 组件：被包裹的 Runnable 调用失败时，按默认重试 3 次的策略自动重试，并配合指数退避，避免重试风暴打垮下游服务；重试范围可通过 `retry_if_exception_type` 参数按异常类型过滤。

截至本篇修订，langchain-core 的当前版本为 0.3.15，发布于 2025-03-12。该版本线由核心维护者陈默负责，重点修复了回调事件在流式模式下的乱序问题；从 0.3.x 的早期补丁升级到 0.3.15 不需要改动业务代码。

观测能力上，core 内置 tracing 机制，设置环境变量即可把每次调用的输入输出上报到追踪后端，配合 `tracing_v2` 采样参数控制上报比例，是排查链式调用问题的第一入口。
