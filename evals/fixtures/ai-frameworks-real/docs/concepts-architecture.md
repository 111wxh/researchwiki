# Architecture

LangChain is a framework that consists of a number of packages.

## langchain-core

This package contains base abstractions for different components and ways to compose them together.
The interfaces for core components like chat models, vector stores, tools and more are defined here.
**No third-party integrations are defined here.** The dependencies are kept purposefully very lightweight.

## langchain

The main `langchain` package contains chains and retrieval strategies that make up an application's cognitive architecture.
These are NOT third-party integrations.
All chains, agents, and retrieval strategies here are NOT specific to any one integration, but rather generic across all integrations.

## Integration packages

Popular integrations have their own packages (e.g. `langchain-openai`, `langchain-anthropic`, etc) so that they can be properly versioned and appropriately lightweight.
