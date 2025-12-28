**Retrieval-Augmented Generation (RAG) Policy Assistant**
Overview: This project implements a Retrieval-Augmented Generation (RAG) system that enables users to query internal documents (such as HR policies or SOPs) and receive accurate, grounded, natural-language answers.

The system runs locally and combines:
1)	Pinecone for document embeddings and vector search
2)	OpenAI for answer generation
3)	FastAPI as the backend service
4)	A lightweight HTML/JavaScript UI for user interaction
5)	Unlike a general chatbot, this application answers only from the ingested documents and does not hallucinate information.

<img width="200" height="609" alt="image" src="https://github.com/user-attachments/assets/487b7bad-ceb6-4a6c-b1bc-0a45c2f5b866" />


<img width="500" height="431" alt="image" src="https://github.com/user-attachments/assets/c6289856-4af7-4e9a-b58c-9e3977e31d3b" />




| Category              | Tool / Technology         | Purpose in This Project                                          |
| --------------------- | ------------------------- | ---------------------------------------------------------------- |
| Programming Language  | **Python 3.10+**          | Core language for ingestion, backend APIs, and integration logic |
| Backend Framework     | **FastAPI**               | Exposes REST APIs for search and generation; serves the UI       |
| ASGI Server           | **Uvicorn**               | Runs the FastAPI application locally                             |
| Vector Database       | **Pinecone**              | Stores embeddings and metadata; performs semantic vector search  |
| Embedding Service     | **Pinecone Inference**    | Generates embeddings for documents and queries                   |
| Embedding Model       | **llama-text-embed-v2**   | Converts text into dense vector representations                  |
| Large Language Model  | **OpenAI**                | Generates natural-language answers                               |
| Generation Model      | **gpt-4o-mini**           | Produces grounded responses from retrieved context               |
| PDF Parsing           | **pypdf**                 | Extracts text from PDF documents                                 |
| DOCX Parsing          | **python-docx**           | Extracts text from Microsoft Word files                          |
| RTF Parsing           | **striprtf**              | Converts RTF documents into plain text                           |
| Text Processing       | **re, hashlib**           | Normalization, hashing, deduplication                            |
| Progress Tracking     | **tqdm**                  | Displays ingestion and embedding progress                        |
| Frontend              | **HTML + JavaScript**     | User interface for interacting with the RAG system               |
| Configuration         | **Environment Variables** | Securely stores API keys and runtime settings                    |
| Dependency Management | **Poetry**                | Manages dependencies and virtual environments                    |


**Design Principles**
1)	No Hallucination: Answers are generated strictly from retrieved context
2)	Explainability: Source documents are returned as citations
3)	Consistency: Same embedding model is used for ingestion and querying
4)	Separation of Concerns: Retrieval and generation are independent layers
5)	Enterprise-Ready: Deterministic, auditable, and extensible design

Localhost -

<img width="1915" height="1037" alt="image" src="https://github.com/user-attachments/assets/0456daaf-814c-4d5d-9141-68c38a8bab1b" />

<img width="1910" height="1051" alt="image" src="https://github.com/user-attachments/assets/2b0b4e8f-9cc1-4a80-860a-3744acefeb67" />



















