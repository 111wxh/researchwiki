# Vector stores

## Overview

Vector stores are specialized data stores that enable indexing and retrieving information based on vector representations.

These vectors, called [embeddings](/docs/concepts/embedding_models/), capture the semantic meaning of data that has been embedded.

Vector stores are frequently used to search over unstructured data, such as text, images, and audio, to retrieve relevant information based on semantic similarity rather than exact keyword matches.

## Interface

LangChain provides a standard interface for working with vector stores, allowing users to easily switch between different vectorstore implementations.

The interface consists of basic methods for writing, deleting and searching for documents in the vector store.

The key methods are:

- `add_documents`: Add a list of texts to the vector store.
- `delete`: Delete a list of documents from the vector store.
- `similarity_search`: Search for similar documents to a given query.
