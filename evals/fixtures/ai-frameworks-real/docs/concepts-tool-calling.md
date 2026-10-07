# Tool calling

## Overview

Many AI applications interact directly with humans. In these cases, it is appropriate for models to respond in natural language.
But what about cases where we want a model to also interact *directly* with systems, such as databases or an API?
These systems often have a particular input schema; for example, APIs frequently have a required payload structure.
This need motivates the concept of *tool calling*. You can use [tool calling](https://platform.openai.com/docs/guides/function-calling/example-use-cases) to request model responses that match a particular schema.

:::info
You will sometimes hear the term `function calling`. We use this term interchangeably with `tool calling`.
:::

:::

The central concept to understand is that LangChain provides a standardized interface for connecting tools to models.
The `.bind_tools()` method can be used to specify which tools are available for a model to call.
