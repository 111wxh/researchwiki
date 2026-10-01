# langchain-retrievers：检索器家族

检索器（Retriever）是 LangChain 中"问题 → 相关文档"的统一接口，所有检索器都实现 `get_relevant_documents` 方法，并可用 `k` 参数控制召回条数，向量库检索器的 k 默认为 4。

ParentDocumentRetriever 解决"小块检索、大块返回"的矛盾：切分时把文档拆成两层，检索命中子块后返回其所属的父文档，保证喂给模型的上下文完整；子块与父块的映射关系存放在后台文档存储中，检索时按映射回溯。

MultiQueryRetriever 针对查询表述单一导致漏召回的问题：默认用 LLM 把原始问题改写成 3 个查询变体，分别检索后合并去重，对措辞敏感的中文查询场景提升明显。

ContextualCompressionRetriever 是装饰器模式：外层负责召回，内层接入 `base_compressor` 压缩器，按与问题的相关性对文档做裁剪或过滤，只把最相关的片段送入提示词，可显著降低 token 消耗。压缩器与检索器各自独立配置，可自由组合。

检索器是纯接口层，可与任意向量库后端组合使用，例如 Milvus、FAISS、Chroma（参见向量库各篇）；更换后端不影响上层管道代码。
