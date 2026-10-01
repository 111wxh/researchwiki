# OpenAI Structured Outputs：严格结构化输出

Structured Outputs 让模型输出严格遵循 JSON Schema：请求时把 `response_format` 设为 `{"type": "json_schema"}`，并在 schema 的每个字段上打开 `strict: true`，服务端即用约束解码保证字段名、类型与枚举完全合法，客户端不再需要"解析失败就重试"的循环。

该能力自 2024-11-15 起对所有支持 JSON 模式的模型开放；不支持 strict 的旧场景可回退 `{"type": "json_object"}`，但后者只保证输出是合法 JSON，并不保证符合给定 schema。

质量评估：在自建抽取基准 FixtureExtract-Bench（受控合成语料口径）上，strict 模式的字段级抽取成功率为 93.7%，明显高于纯提示词约束的 78% 左右；剩余失败集中在超深层嵌套 schema 的场景。

使用限制：strict 模式首次使用新 schema 时需要预热，首 token 延迟更高；`additionalProperties` 必须显式设为 false；所有字段必须标记 required，可选语义用 union 类型表达。配合函数调用（Function Calling）时，同一套 schema 语法可用于 `tools` 参数的定义，实现"调用即校验"。
