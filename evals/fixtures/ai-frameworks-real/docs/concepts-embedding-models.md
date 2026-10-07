# Embedding models

Imagine being able to capture the essence of any text - a tweet, document, or book - in a single, compact representation.
This is the power of embedding models, which lie at the heart of many retrieval systems.
Embedding models transform human language into a format that machines can understand and compare with speed and accuracy.
These models take text as input and produce a fixed-length array of numbers, a numerical fingerprint of the text's semantic meaning.

LangChain provides a universal interface for working with them, providing standard methods for common operations.
This common interface simplifies interaction with various embedding providers through two central methods:

- `embed_documents`: For embedding multiple texts (documents)
- `embed_query`: For embedding a single text (query)

This distinction is important, as some providers employ different embedding strategies for documents (which are to be searched) versus queries (the search input itself).
