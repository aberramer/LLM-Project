# 🚀 LLM Project

This project provides a **Large Language Model (LLM)** powered service.  
It exposes REST API endpoints built with **FastAPI** for tasks like sentiment analysis, text classification, and other NLP use cases.

## 📌 Features
- 🔹 FastAPI-based REST API
- 🔹 `litellm` integration for LLM calls
- 🔹 Prometheus metrics support
- 🔹 Rate limiting & retry/backoff mechanism
- 🔹 Configurable via `config.yaml`
- 🔹 Bearer token authentication

## ⚙️ Installation

### Requirements
- Python 3.10+
- pip

### Steps
```bash
# Clone the repository
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
