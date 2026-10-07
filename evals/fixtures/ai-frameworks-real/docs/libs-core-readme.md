# 🦜🍎️ LangChain Core

## Quick Install

```bash
pip install langchain-core
```

## What is it?

LangChain Core contains the base abstractions that power the the LangChain ecosystem.

These abstractions are designed to be as modular and simple as possible.

For full documentation see the [API reference](https://python.langchain.com/api_reference/core/index.html).

## 📕 Releases & Versioning

As `langchain-core` contains the base abstractions and runtime for the whole LangChain ecosystem, we will communicate any breaking changes with advance notice and version bumps. The exception for this is anything in `langchain_core.beta`. The reason for `langchain_core.beta` is that given the rate of change of the field, being able to move quickly is still a priority, and this module is our attempt to do so.

Minor version increases will occur for:

- Breaking changes for any public interfaces NOT in `langchain_core.beta`
