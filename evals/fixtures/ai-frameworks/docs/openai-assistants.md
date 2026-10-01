# OpenAI Assistants API：托管智能体

Assistants API 是 OpenAI 的托管智能体服务：开发者创建 Assistant 对象，把模型、指令与工具绑定在一起，由服务端负责工具调用循环，客户端不必自己写 ReAct 逻辑。该 API 的 v2 版本于 2024-08-20 发布，引入向量存储（Vector Storage）对象后，文件检索不再需要自建向量库。

核心模型是 Thread 与 Run：Thread 是与用户的会话容器，消息持续追加其中；Run 把 Assistant 绑定到某个 Thread 上异步执行，客户端轮询 run 状态直到 completed，再拉取新增消息。

工具方面，每个 Assistant 最多可挂载 32 个工具，类型包括代码解释器、文件检索与自定义函数。`tool_choice` 参数控制工具调用策略：设为 "auto" 时由模型自行决定每次是否调用工具，也可以强制指定调用某个特定工具。

Run 级别还有并行工具调用开关 `parallel_tool_calls`，默认开启；对有先后依赖的工具序列应当关闭，避免乱序写操作。

与自研循环（如 LangChain Agents）相比，托管方案省去运维但灵活性低；多智能体编排、复杂状态转移仍需外部框架补位（参见多框架对比篇）。
