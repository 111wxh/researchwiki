# Structured outputs

## Overview

For many applications, such as chatbots, models need to respond to users directly in natural language.
However, there are scenarios where we need models to output in a *structured format*.
For example, we might want to store the model output in a database and ensure that the output conforms to the database schema.
This need motivates the concept of structured output, where models can be instructed to respond with a particular output structure.

## Recommended usage

This pseudocode illustrates the recommended workflow when using structured output.
LangChain provides a method, [`with_structured_output()`](/docs/how_to/structured_output/#the-with_structured_output-method), that automates the process of binding the schema to the [model](/docs/concepts/chat_models/) and parsing the output.
This helper function is available for all model providers that support structured output.
