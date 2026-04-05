# 📚 Local Agentic RAG Chatbot (Ollama + Chroma + Streamlit)

A fully local Retrieval-Augmented Generation (RAG) chatbot built with:

- 🧠 Ollama (local LLM + embeddings)
- 🗄 ChromaDB (vector store)
- 🔗 LangChain (Classic + Core)
- 🖥 Streamlit UI
- 🐍 Python 3.12

This project ingests scraped Wikipedia content, embeds it locally, stores it in Chroma, and provides an agentic chat interface for querying that data.

No OpenAI. No cloud inference. Fully local.

---

## 🎥 Reference Tutorial

This project is based on / inspired by the following YouTube tutorial:

https://www.youtube.com/watch?v=c5jHhMXmXyo

⚠️ Note: The tutorial uses older LangChain APIs. This repo has been modernized to work with:

- langchain-core  
- langchain-classic  
- langchain-ollama  
- langchain-chroma  

and includes defensive fixes for tool calling + Pydantic validation.

---

## ✨ Features

- Local LLM inference via Ollama
- Local vector database (Chroma)
- Agent-based retrieval
- Streamlit chat UI
- Source attribution
- Defensive tool handling (robust against malformed agent calls)
- Fully reproducible Python environment

---

## 📁 Project Structure
LOCAL-RAG-WITH-OLLAMA/
│
├── chroma_db/                     # Persistent Chroma vector database
├── datasets/                      # Scraped / downloaded raw datasets
├── venv/                          # Python virtual environment (not committed)
│
├── .env                           # Environment variables (not committed)
├── .gitignore
│
├── 1_scraping_wikipedia.py        # Scrapes Wikipedia using BrightData
├── 2_chunking_embedding_ingestion.py
│                                 # Chunks text, generates embeddings,
│                                 # and ingests into ChromaDB
├── 3_chatbot.py                   # Streamlit Agentic RAG chatbot UI
│
├── example_chunking.py            # Standalone chunking example
├── example_embedding.py           # Standalone embedding example
├── example_retriever.py           # Standalone retrieval example
│
├── keywords.xlsx                  # Input keywords for scraping
├── snapshot.txt                   # BrightData snapshot tracking
│
├── thumbnail_small.png            # Project thumbnail / reference image
│
├── requirements.txt               # Frozen Python dependencies
└── README.md                      # Project documentation

