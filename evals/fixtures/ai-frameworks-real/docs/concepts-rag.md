# Retrieval augmented generation (RAG)

## Overview

Retrieval Augmented Generation (RAG) is a powerful technique that enhances [language models](/docs/concepts/chat_models/) by combining them with external knowledge bases.
RAG addresses [a key limitation of models](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise): models rely on fixed training datasets, which can lead to outdated or incomplete information.
When given a query, RAG systems first search a knowledge base for relevant information.
The system then incorporates this retrieved information into the model's prompt.
The model uses the provided context to generate a response to the query.

(1) **Retrieval system**: Retrieve relevant information from a knowledge base.

(2) **Adding external knowledge**: Pass retrieved information to a model.
