# Text splitters

## Overview

Document splitting is often a crucial preprocessing step for many applications.
It involves breaking down large texts into smaller, manageable chunks.
This process offers several benefits, such as ensuring consistent processing of varying document lengths, overcoming input size limitations of models, and improving the quality of text representations used in retrieval systems.
There are several strategies for splitting documents, each with its own advantages.

## Why split documents?

### Length-based

The most intuitive strategy is to split documents based on their length. This simple yet effective approach ensures that each chunk doesn't exceed a specified size limit.
Key benefits of length-based splitting:
- Straightforward implementation
- Consistent chunk sizes
- Easily adaptable to different model requirements
