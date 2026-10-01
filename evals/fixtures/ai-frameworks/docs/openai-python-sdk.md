# OpenAI Python SDK：客户端基础

OpenAI Python SDK 是访问 OpenAI API 的官方客户端，提供聊天补全、嵌入、文件与批处理等资源方法。SDK 遵循语义化版本号，本篇口径为 1.61.1，发布于 2025-02-05。

可靠性方面有两个关键参数：`max_retries` 控制请求失败后的自动重试次数，默认 2 次，遇到 429 限流或 5xx 服务端错误时按指数退避重试；`timeout` 控制单个请求的超时时间，默认 600 秒，长文本批量任务建议显式调小，以免调用方被单个慢请求挂住。

SDK 内置类型化的异常分层：认证失败抛 AuthenticationError，限流抛 RateLimitError，参数问题抛 BadRequestError，业务代码可以按异常类型精细化处理，而不必解析响应体字符串。

流式输出通过 `stream=True` 开启，返回事件迭代器；结构化输出配合 `response_format` 参数使用（详见结构化输出篇）。每次响应都带 usage 字段，记录输入与输出 token 数，便于计费对账；批处理接口按半价计费，适合离线跑全量任务。
