 **Project Goal / Problem Statement**
Organizations often store critical knowledge across multiple unstructured sources such as PDFs, text files, and JSON documents.
Traditional keyword search systems struggle to retrieve contextually relevant information, especially when documents are large, duplicated across formats, or distributed across multiple domains (e.g., drug documentation).

**The goal of this project is to design and implement an end-to-end Retrieval-Augmented Generation (RAG) system that**:
1.	Ingests unstructured documents from multiple formats
2.	Performs semantic search instead of keyword search
3.	Generates accurate, context-grounded answers using an LLM
4.	Scales in a production-style, API-driven architecture


**High-Level Solution**: This project implements a RAG pipeline where

1.	Documents are converted into vector embeddings.
2.	Vectors are stored in a scalable vector database (Pinecone)
3.	User queries are semantically matched against stored vectors
4.	Relevant context is retrieved and passed to an LLM
5.	The LLM generates answers grounded strictly in retrieved content

**Tools & Technologies Used**
Backend & API
FastAPI – High-performance API framework
Uvicorn – ASGI server
Pydantic – Request/response validation
Vector Database
Pinecone – Managed vector database for scalable similarity search
Namespaces & metadata filtering – for domain-specific retrieval
Embeddings & LLM
Pinecone Integrated Embeddings (llama-text-embed-v2, 1024-dim)
OpenAI GPT models – for grounded answer generation

**Data Processing**
Python
PDF parsing (pypdf)
Text normalization & chunking
Deduplication logic
Poetry – Dependency management
Git & GitHub – Version control
Environment variables – Secure secret handling

**1)Data Ingestion**
Accepted documents in PDF, TXT, and JSON formats
Normalized text content across all formats
Extracted metadata (e.g., drug_id, doc_type, source files)

**2)Deduplication**
Documents appearing in multiple formats were deduplicated using content hashes
Prevented redundant vector storage and reduced cost

**3)Chunking**
Large documents split into overlapping text chunks
Optimized chunk size for embedding quality and retrieval accuracy

 **4)Vector Embedding**
Used Pinecone’s integrated embedding model
Ensured embedding dimension matched the Pinecone index (1024-dim)

**5)Vector Storage**
Stored vectors in Pinecone with rich metadata:
drug_id
doc_type
source file references
Enabled metadata-filtered retrieval

**6)Query Processing**
User query embedded using the same embedding model
Performed semantic similarity search
Retrieved top-K most relevant chunks

**7)Answer Generation (RAG)**
Retrieved chunks assembled into a context window
OpenAI LLM generates an answer only using retrieved context
Returned:
Generated answer
Citations (source metadata & scores)

**8)API Layer**
Exposed functionality via REST APIs:
/search – Semantic retrieval only
/generate – Retrieval + LLM answer generation
/health – Service health check

**Outcome / Results**
Successfully built a production-style RAG system
Enabled accurate semantic search across unstructured documents
Prevented hallucinations by enforcing context-only generation


**Possible Future Enhancements**

1.	Add authentication & role-based access
2.	Support streaming responses
3.	Introduce hybrid search (keyword + vector)
4.	Add document versioning
5.	Integrate frontend UI (React/Next.js)
6.	Deploy using Docker/Kubernetes

FastAPI -

<img width="911" height="385" alt="image" src="https://github.com/user-attachments/assets/45be0303-ade3-462e-b493-72bac67434b8" />






