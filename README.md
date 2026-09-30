\# Smart Meter AI Assistant



A local Retrieval-Augmented Generation (RAG) knowledge assistant designed for smart meter operations.



The system allows users to ask questions about smart meter operations and retrieves relevant information from a curated knowledge base of synthetic smart-meter business documents.



\## Project Overview



The Smart Meter AI Assistant combines:



\- Document ingestion

\- Text cleaning

\- Document chunking

\- Sentence embeddings

\- FAISS vector search

\- Retrieval-based context selection

\- Local LLM generation using Ollama

\- FastAPI backend

\- Web-based frontend

\- Source-aware responses



\## Architecture



User

&#x20; ↓

Web Frontend

&#x20; ↓

FastAPI

&#x20; ↓

RAG Pipeline

&#x20; ↓

Query Embedding

&#x20; ↓

FAISS Vector Search

&#x20; ↓

Top-K Relevant Chunks

&#x20; ↓

Prompt Construction

&#x20; ↓

Local LLM (Ollama)

&#x20; ↓

Answer + Source Citation

&#x20; ↓

Frontend

