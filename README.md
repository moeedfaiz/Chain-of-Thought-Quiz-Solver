# Chain-of-Thought-Quiz-Solver

🧠 LangGraph-Powered Reasoning Assistant with RAG
This project is an interactive Reasoning and MCQ Assistant that leverages LangGraph, LangChain, and FAISS-based Retrieval-Augmented Generation (RAG) to process, decompose, and answer complex reasoning-style questions. It supports step-by-step logical reasoning, context retrieval, and even memory of past Q&A — all served through a simple web interface built with Flask.

✅ Key Features
LangGraph Integration: Uses a multi-node graph pipeline (decomposition → retrieval → answering) for modular and interpretable reasoning.

Document RAG: Loads MCQs with step-by-step solutions from a JSON file and creates a vector index using FAISS and HuggingFace embeddings.

LLM-Powered Decomposition: Questions are first broken down into smaller logical steps by an LLM using a specialized prompt.

Memory Recall: Remembers previous answers across sessions and can recall them when asked ("What was the previous answer?").

Flask Web UI: Offers a user-friendly web interface for asking questions, viewing answers, and tracking memory history.

Robust Error Handling: Built-in retry mechanism for API failures with exponential backoff.

🧩 How It Works
User asks a question → Sent to the LangGraph pipeline.

Decomposition Node → Breaks down complex questions into logical steps using an LLM.

Retriever Node → Fetches the most relevant context from the preloaded MCQs using FAISS and MiniLM embeddings.

Answer Node → Uses another LLM prompt to answer the question step-by-step with context.

Memory System → Stores Q&A pairs and recalls them when asked.

🛠️ Tech Stack
LangGraph – Manages reasoning as a directed graph of nodes

LangChain – Handles LLMs, prompts, embeddings, vector stores

Groq + LLaMA 3 – For ultra-fast and cost-effective LLM inference

FAISS – Fast vector similarity search for retrieval

HuggingFace Embeddings – For semantic similarity (MiniLM)

Flask – Lightweight web server and interface

HTML/JS – Simple frontend with memory logging