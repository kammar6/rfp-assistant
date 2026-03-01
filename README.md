# Privacy-First RFP Assistant (RAG Pipeline)

An automated, locally-hosted Retrieval-Augmented Generation (RAG) pipeline designed to answer B2B Requests for Proposal (RFPs). 

This project ensures strict data privacy by running all machine learning models and vector databases locally, preventing sensitive corporate documents from being transmitted to external APIs.

## Architecture & Tech Stack
This application is built as a microservice architecture:

* **Orchestration:** [n8n](https://n8n.io/) (running in Docker) handles event routing and triggers.
* **Backend Logic:** Python / FastAPI processes incoming documents, performs text extraction (via PyMuPDF), and manages text chunking.
* **Vector Database:** [Qdrant](https://qdrant.tech/) (running in Docker) stores text embeddings for semantic search.
* **Local AI/ML:** [Ollama](https://ollama.com/) runs directly on the Ubuntu host, utilizing `nomic-embed-text` for vectorization and local LLMs for response generation.

## Current Data Flow
1. A PDF document is submitted via an n8n webhook.
2. The FastAPI backend extracts the text and splits it into overlapping chunks to preserve context.
3. The chunks are sent to the local Ollama instance to generate 768-dimensional mathematical vectors.
4. The vectors and metadata are stored in Qdrant for future similarity searches.


### Prerequisites
* Ubuntu / Linux environment
* Docker & Docker Compose
* Python 3.10+
* Ollama installed natively on the host
