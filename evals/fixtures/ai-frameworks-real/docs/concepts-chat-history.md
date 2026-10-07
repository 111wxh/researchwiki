# Chat history

Chat history is a record of the conversation between the user and the chat model. It is used to maintain context and state throughout the conversation. The chat history is sequence of [messages](/docs/concepts/messages), each of which is associated with a specific [role](/docs/concepts/messages#role), such as "user", "assistant", "system", or "tool".

Most conversations start with a **system message** that sets the context for the conversation. This is followed by a **user message** containing the user's input, and then an **assistant message** containing the model's response.

The **assistant** may respond directly to the user or if configured with tools request that a [tool](/docs/concepts/tool_calling) be invoked to perform a specific task.

## Managing chat history

Since chat models have a maximum limit on input size, it's important to manage chat history and trim it as needed to avoid exceeding the [context window](/docs/concepts/chat_models/#context-window).
