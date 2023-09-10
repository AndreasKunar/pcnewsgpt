# Ideas/Concepts for Improvements

Collecting Technology-Ideas for for PCnewsGPT Improvements

## Document Sections: Better rendering of chunks for long documents

[Source](https://community.openai.com/t/document-sections-better-rendering-of-chunks-for-long-documents/329066)

ToDo (complicated)

## How to Chunk Text Data A Comparative Analysis

[Source](https://towardsdatascience.com/how-to-chunk-text-data-a-comparative-analysis-3858c4a0997a)

### NLTK Sentence Tokenizer and Spacy Sentence Splitter

These approaches exhibit a preference for smaller sentences but include many larger outliers. While this can result in more linguistically coherent chunks, it can also lead to high variability in chunk size.

Spacy strictly adheres to sentence boundaries. This can be advantageous when smaller text units are necessary for analysis. Spacy’s performance depends on the quality of the input text. For poorly punctuated or structured text, the identified sentence boundaries might not always be accurate.

These methods can yield good results that can serve as inputs to downstream tasks too.

### Adjacent Sequence Clustering

This method generates a more varied distribution, indicative of its context-sensitive approach. By clustering based on semantic similarity, it ensures that the content within each chunk is coherent while allowing for flexibility in chunk size. This method may be advantageous when it is important to preserve the semantic continuity of text data.

## Effects of Chunk Sizes on Retrieval Augmented Generation (RAG) Applications

[Source](https://reframe.is/wiki/Effects-of-Chunk-Sizes-on-Retrieval-Augmented-Generation-RAG-Applications-8b728c36d005434dba39ad19be9b82cc/?pvs=4)

### Effects of Chunk Sizes on RAG

The retrieval module is designed to find the most relevant passages from a large collection of texts to augment the context for the language model. Operating on longer chunks of text allows the retriever to identify documents with broad topical relevance even when matches may be approximate. In contrast, the language model benefits from conditioning on shorter, more coherent spans during synthesis so it can maintain consistency in generated text.
Feeding the language model full retrieved passages verbatim can introduce repetition or even contradictions. By extracting only the most salient phrases or sentences from a longer retrieved chunk, the model is less likely to diverge or hallucinate. Separating the chunking strategies thus plays to the strengths of each module - efficient passage retrieval versus controlled generation. The retriever provides overall contextual background, while salient details are judiciously filtered to incrementally guide the language model's word predictions through a focused contextual lens. In this way, keeping distinct chunk sizes can improve the stability and accuracy of retrieval-augmented generation.
Text Chunking Strategies for RAG

### Heirarchical Storage for RAG

Storing data hierarchically is an optimization technique that can improve the performance of the retrieval module in RAG models. Specifically, it involves structuring the knowledge source documents into multiple layers:

+ Top level - concise summaries or titles for each document
+ Middle level - key passages or chunks extracted from each document
+ Bottom level - the full documents

With this structure, retrieval can happen much more efficiently:

+ The model first searches the top layer summaries to select the most relevant documents broadly. This is faster than scanning all documents.
+ It then searches the passages of just the selected documents to find the most pertinent chunks.
+ Only a few full documents need to be processed in detail.

Additionally, the summaries provide a brief overview of the context, while passages offer more focused, salient details.
Hierarchical storage provides a multi-resolution view of the knowledge that allows retrieving general context, key details, or full background flexibly as needed.
Some other benefits include:

+ Faster nearest-neighbor search for relevant content
+ Reduced memory usage by avoiding loading full documents
+ Ability to select granularity of context based on prompt
+ Improved consistency by conditioning on focused passages

In summary, storing data hierarchically aligns well with the multi-step nature of retrieval in RAG models. It enables efficient and targeted extraction of relevant context at different levels of specificity to augment the language model prompts.

### Problems with RAG nor providing relevant Contexts for the LLM

If the retrieval module in a RAG model is not providing relevant context for the language model, it usually indicates an issue with the knowledge sources it is searching. There are a few potential reasons why the retrieved passages may not match the prompt:

+ The document collection is too small or limited in scope. Expanding the size and diversity of sources like Wikipedia articles, papers, etc. can increase chances of finding good matches.
+ The existing documents are outdated and don't contain information on recent developments. Static corpora can go stale over time.
+ The embedding space used for retrieval is a poor fit for the prompt domain. Different embedding approaches work better for some topics.
+ There is a vocabulary mismatch between the prompts and documents. Similar words may have different meanings.
+ The retrieval algorithm itself needs tuning to better match prompts to passage content/context.

### Expanding LLM Prompt Contexts

Here are a few ways an engineer could provide more context to a large language model like RAG when generating text, to avoid issues with insufficient context:

+ Store document chunks in a data structure like a bidirectional Linked List list that maintains order and relationships between chunks. When retrieving a chunk to send to the LLM, also retrieve the preceding and following chunks in the list to provide more surrounding context.
+ Simply store the preceding and next sentence within the vector index like Pinecone that allows retrieving all the information needed for the LLM prompt in a single shot. The drawback to this that it dramatically increases your storage requirements by 3x.
+ Store metadata on the relationships between document chunks - like which chunks are from the same document, which chunks mention the same entities etc. When retrieving a chunk, also query for and retrieve related chunks using this metadata to provide more relevant context.
+ Maintain sliding windows of text from documents/chunks and pass these windows rather than isolated chunks to the LLM. The sliding windows preserve local context even when retrieving specific chunks. The window size can be tuned based on how much context is needed.
+ Use an encoder model to encode chunks/documents into latent representations. Retrieve chunks by proximity in the latent space rather than keyword. Latent representation matches may have more semantic relationships than keyword matches, providing more useful context.

The core idea is to leverage the structure of the external knowledge storage to not only retrieve isolated fragments for the LLM but also surrounding context from related fragments that can ground the LLM's predictions. Linked lists, metadata, vector similarity, and windowing are all techniques to achieve this.