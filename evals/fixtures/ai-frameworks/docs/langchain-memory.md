# langchain-memory：对话记忆组件

langchain-memory 为链式调用提供会话记忆，把历史对话注入提示词。最基础的 ConversationBufferMemory 原样保留全部历史；对话变长后上下文占用迅速膨胀，因此库内提供三种裁剪策略。

ConversationBufferWindowMemory 只保留最近 k 轮对话，默认 k=5，超出窗口的早期内容被直接丢弃，适合闲聊类短任务。ConversationSummaryMemory 用 LLM 对历史做滚动摘要，每轮结束后把新对话合并进摘要，上下文占用恒定，但会丢失细节。ConversationSummaryBufferMemory 是两者折中：摘要保底，最近若干条消息保留原文。

记忆注入链的方式通过 `memory_key` 参数指定，默认注入键名为 chat_history，必须与提示词模板中的占位符同名，否则运行时报变量缺失错误。

多用户场景下，每个会话应使用独立的 session_id 隔离记忆，避免不同用户的上下文串话。记忆对象本身不持久化，进程重启即丢失；跨会话的长期存储需要外接向量库或数据库，相关设计取舍参见记忆系统设计篇。
