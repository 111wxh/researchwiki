# OpenAI SDK 与 API 版本史

OpenAI Python SDK 在 2023-11-06 发布 1.0.0：这次大版本全面重写，从带全局状态的旧客户端改为实例化客户端，所有资源方法挂到 client 对象之下，旧的全局写法自此废弃。

1.40.0 于 2024-09-10 发布，重构了异常类型分层，并把流式事件统一为带类型判别字段的事件对象，SDK 侧错误语义自此与 API 对齐。

API 侧两条主线：Assistants API v2 于 2024-08-20 发布，新增向量存储对象与并行工具调用开关；Structured Outputs 于 2024-11-15 全量开放，`response_format` 的 json_schema 模式开始提供严格保证。两者发布顺序上，Assistants v2 早于 Structured Outputs 约三个月。

本篇只记历史主干，不追记最新补丁；SDK 当前版本与默认参数的最新口径，以 openai-python-sdk 文档（含修订版）为准。

版本兼容提示：1.x 系列内保持向后兼容，跨大版本升级前应查阅迁移指南；服务端 API 版本通过请求头固定，SDK 升级不改变已固定的 API 版本。
