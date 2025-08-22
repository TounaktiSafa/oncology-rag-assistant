# Medical Oncology RAG System
A specialized Retrieval-Augmented Generation (RAG) system for oncology information, providing accurate medical answers from curated PDF documents.

# Features
Document Processing: Extracts and chunks text from medical PDFs

Specialized Embeddings: Uses PubMedBERT for medical text understanding

Vector Database: Stores document embeddings in Qdrant for efficient retrieval

Medical Q&A: Provides accurate answers with source references

Web Interface: User-friendly interface for querying medical information

# Prerequisites
Python 3.8+

Qdrant database (running via Docker)

Medical PDF documents in the Data/ folder
