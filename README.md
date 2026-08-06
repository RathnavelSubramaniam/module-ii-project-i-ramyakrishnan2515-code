# **Enhancing LLM Accuracy in Healthcare using RAG and Automated Evaluation**


### Introduction

Modern clinical practice generates an overwhelming quantity of medical literature, treatment guidelines, and diagnostic updates. Healthcare professionals face severe time constraints and cognitive fatigue when retrieving precise clinical information at the point of care. 

While Large Language Models (LLMs) offer strong natural language processing capabilities, relying solely on parametric memory poses risks of medical hallucinations, unverified factual claims, and out-of-date information. Implementing a Retrieval-Augmented Generation (RAG) framework anchors LLM outputs in verified, domain-specific reference materials to provide accurate, evidence-based decision support.

### Project Overview

This project presents an end-to-end medical RAG pipeline integrated with automated evaluation metrics. 

By pairing dense semantic retrieval across authoritative clinical texts with local quantized model inference, the architecture mitigates hallucination risks and enhances response reliability.

- Automate Medical Ingestion & Indexing: Parse, chunk, and embed multi-thousand-page medical literature (4,000+ pages across 23 clinical sections) into ChromaDB for rapid semantic context retrieval.

- Mitigate LLM Hallucinations: Enforce strict grounding of generated responses within retrieved medical passages to eliminate factual inaccuracies.

- Accelerate Point-of-Care Retrieval: Synthesize complex clinical guidelines into real-time, context-aware summaries.

- Automated & Rigorous Evaluation: Benchmark RAG-generated answers against standard prompt-engineered LLM responses using automated metrics for Groundedness and Answer Relevance.

### Dataset Overview

* Source Material: The Merck Manual of Diagnosis & Therapy (19th Edition) (~4,114 pages).

* Chunking Strategy: RecursiveCharacterTextSplitter using Tiktoken encoding with chunk_size=520 tokens and chunk_overlap=60 tokens (resulting in ~8,757 processed chunks).

* Embeddings: sentence-transformers/all-MiniLM-L6-v2 generating normalized dense vector representations.

* Vector Store: Persistent ChromaDB index configured for fast cosine similarity retrieval.

### Install Libraries

To set up the project environment, begin by removing any pre-existing NumPy packages and installing NumPy version 1.26.4 to ensure strict C-extension compatibility across dependencies.  

Next, install the core orchestration, vector storage, and evaluation dependencies:  

- LangChain and LangChain Community (version 0.3.27) for RAG orchestration and vector store adapters. 
- ChromaDB (version 1.0.15) for vector index persistence.
- PyMuPDF (version 1.26.3) for PDF parsing and extraction.
- Tiktoken (version 0.9.0) for token-based text chunking.
- Datasets (version 4.0.0) and Evaluate (version 0.4.5) for automated model evaluation.
- LangChain OpenAI (version 0.3.30), LangChain HuggingFace, Sentence-Transformers, HuggingFace Hub, and Transformers for embedding generation and repository management.

Finally, build and install llama-cpp-python (version 0.2.45) with CUDA acceleration enabled by setting the CMake flags CMAKE_ARGS="-DLLAMA_CUBLAS=on" and FORCE_CMAKE=1.  

### Conclusion

- Integrating dense semantic retrieval via ChromaDB with quantized LLaMA-2 inference substantially reduces hallucination risks compared to parametric generation. 

- Constraining the language model strictly to evidence retrieved from verified medical texts (Merck Manual) improves answer accuracy, groundedness, and domain relevance for clinical decision support.

### Future Recommendations

- Hybrid Retrieval: Combine dense vector search with sparse keyword search (BM25) to capture specific clinical terminology and dosage values more accurately.

- Hierarchical Chunking: Implement parent-document retrieval to maintain broad diagnostic context while retrieving precise sub-paragraphs.

- Citation Integration: Update prompt structures to require precise in-text page/section citations for direct clinical auditing.

- Continuous Evaluation Pipeline: Implement automated benchmarking on live clinical query datasets using framework metrics like Ragas or TruLens.
