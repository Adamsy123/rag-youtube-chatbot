# 🎥🤖 RAG YouTube Chatbot & Media Downloader

An all-in-one Streamlit application that allows users to download YouTube media in custom formats/resolutions and engage in Context-Aware RAG (Retrieval-Augmented Generation) chat with the video transcript powered by OpenAI and FAISS.

---

## ✨ Features

* **High-Quality Video Muxing:** Automatically handles YouTube's adaptive video and audio separation by combining streams using `imageio-ffmpeg` without quality loss.
* **Audio Extraction:** Option to directly save videos as clean `.mp3` files.
* **Dual Download Modes:** Save files directly to your system's `Downloads` folder or retrieve them through the Streamlit web interface.
* **RAG Chat Engine:** Ingests video closed-captions, indexes them using `FAISS` vector stores, and answers context-specific questions via `GPT-4o-mini`.

---

## 🛠️ Architecture Overview

1. **Extraction:** `pytubefix` fetches progressive and adaptive streams from YouTube.
2. **Audio/Video Stitching:** `imageio-ffmpeg` executes lossless muxing for high-resolution video fallback streams.
3. **Ingestion & Embedding:** Transcripts are fetched via `YoutubeLoader`, chunked using `RecursiveCharacterTextSplitter`, and converted to vector space via `OpenAIEmbeddings`.
4. **Retrieval & Generation:** A `FAISS` vector retriever passes relevant context chunks into a standard LangChain RAG pipeline.

---

## 🚀 Quickstart

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/rag-youtube-chatbot.git](https://github.com/YOUR_GITHUB_USERNAME/rag-youtube-chatbot.git)
cd rag-youtube-chatbot
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory and add your OpenAI API Key:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 4. Running the App
Start the Streamlit interface:

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```plaintext
├── app.py              # Main application code (Streamlit UI + RAG Engine + Muxing)
├── requirements.txt    # Project dependencies
├── .env.example        # Template for key configuration
├── .gitignore          # Rules for excluded files from git
└── README.md           # Documentation
```