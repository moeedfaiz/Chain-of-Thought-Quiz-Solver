# import os
# import json
# import time
# from typing import List, Dict
# from dotenv import load_dotenv
# from cohere.errors import TooManyRequestsError
# from langchain_cohere import ChatCohere, CohereEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document
# from langgraph.graph import StateGraph, END

# # ========= Step 1: Load environment and API key =========
# load_dotenv()
# if not os.getenv("COHERE_API_KEY"):
#     raise ValueError("COHERE_API_KEY not found in .env file")

# # ========= Step 2: Initialize LLM and Embedding =========
# llm = ChatCohere(model="command-r", temperature=0.3)
# embedding_model = CohereEmbeddings(model="embed-english-v3.0")

# # ========= Step 3: Load and split documents =========
# def load_data() -> List[Document]:
#     with open("mcqs.json", "r", encoding="utf-8") as f:
#         raw = json.load(f)
#     texts = [
#         f"Question: {item['question']}\nSteps:\n" + "\n".join(item['steps']) + f"\nAnswer: {item['answer']}"
#         for item in raw if 'question' in item and 'steps' in item and 'answer' in item
#     ]
#     splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
#     docs = splitter.create_documents(texts)
#     return docs

# documents = load_data()

# # ========= Step 4: Create Vector Store =========
# vectorstore = FAISS.from_documents(documents, embedding_model)
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # ========= Step 5: Prompt Templates =========
# decompose_prompt = PromptTemplate.from_template(
#     "Decompose this complex question into smaller reasoning steps:\n\nQuestion: {question}\n\nSteps:"
# )

# answer_prompt = PromptTemplate.from_template(
#     "Given the following context:\n\n{context}\n\nAnswer this question step-by-step:\n{question}"
# )

# # ========= Step 6: Define Graph State =========
# class GraphState(Dict):
#     question: str
#     steps: str
#     context: str
#     answer: str

# # ========= Step 7: Safe LLM Call =========
# def safe_llm_invoke(prompt: str) -> str:
#     retries = 3
#     for attempt in range(retries):
#         try:
#             response = llm.invoke(prompt)
#             return response.content
#         except TooManyRequestsError:
#             wait_time = 10 * (attempt + 1)
#             print(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds...")
#             time.sleep(wait_time)
#         except Exception as e:
#             print("❌ LLM invocation failed:", str(e))
#             break
#     raise Exception("LLM call failed after retries.")

# # ========= Step 8: Graph Nodes =========
# def decompose_question(state: GraphState) -> GraphState:
#     response = safe_llm_invoke(decompose_prompt.format(question=state["question"]))
#     return {"question": state["question"], "steps": response}

# def retrieve_context(state: GraphState) -> GraphState:
#     docs = retriever.get_relevant_documents(state["steps"])
#     context_text = "\n".join([doc.page_content for doc in docs])
#     return {
#         "question": state["question"],
#         "steps": state["steps"],
#         "context": context_text,
#     }

# def answer_question(state: GraphState) -> GraphState:
#     response = safe_llm_invoke(
#         answer_prompt.format(context=state["context"], question=state["question"])
#     )
#     return {
#         "question": state["question"],
#         "steps": state["steps"],
#         "context": state["context"],
#         "answer": response,
#     }

# # ========= Step 9: Build Graph =========
# workflow = StateGraph(GraphState)
# workflow.add_node("Decompose", decompose_question)
# workflow.add_node("Retrieve", retrieve_context)
# workflow.add_node("Answer", answer_question)

# workflow.set_entry_point("Decompose")
# workflow.add_edge("Decompose", "Retrieve")
# workflow.add_edge("Retrieve", "Answer")
# workflow.add_edge("Answer", END)

# app = workflow.compile()

# # ========= Step 10: Run App =========
# if __name__ == "__main__":
#     try:
#         user_question = input("Ask your complex MCQ or reasoning question:\n> ")
#         result = app.invoke({"question": user_question})
#         print("\n🧠 Final Answer:\n", result["answer"])
#     except Exception as e:
#         print("❌ Error:", str(e))




import os
import json
import time
from typing import List, Dict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# ========= Step 1: Load environment and API key =========
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# ========= Step 2: Initialize LLM and Embedding =========
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192",
    temperature=0.3
)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ========= Step 3: Load and split documents =========
def load_data() -> List[Document]:
    with open("mcqs.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    texts = [
        f"Question: {item['question']}\nSteps:\n" + "\n".join(item['steps']) + f"\nAnswer: {item['answer']}"
        for item in raw if 'question' in item and 'steps' in item and 'answer' in item
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents(texts)
    return docs

documents = load_data()

# ========= Step 4: Create Vector Store =========
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ========= Step 5: Prompt Templates =========
decompose_prompt = PromptTemplate.from_template(
    "Decompose this complex question into smaller reasoning steps:\n\nQuestion: {question}\n\nSteps:"
)

answer_prompt = PromptTemplate.from_template(
    "Given the following context:\n\n{context}\n\nAnswer this question step-by-step:\n{question}"
)

# ========= Step 6: Define Graph State =========
class GraphState(Dict):
    question: str
    steps: str
    context: str
    answer: str
    memory: List[Dict[str, str]]  # For tracking past answers

# ========= Step 7: Safe LLM Call =========
def safe_llm_invoke(prompt: str) -> str:
    retries = 3
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            wait_time = 10 * (attempt + 1)
            print(f"\u26a0\ufe0f Error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    raise Exception("LLM call failed after retries.")

# ========= Step 8: Graph Nodes =========
def decompose_question(state: GraphState) -> GraphState:
    question = state["question"]

    # If asking for last answer:
    if "previous answer" in question.lower() or "last answer" in question.lower():
        memory = state.get("memory", [])
        if memory:
            return {"question": question, "steps": "", "context": "", "answer": memory[-1]['answer'], "memory": memory}
        else:
            return {"question": question, "steps": "", "context": "", "answer": "No previous answer available.", "memory": []}

    response = safe_llm_invoke(decompose_prompt.format(question=question))
    return {"question": question, "steps": response, "memory": state.get("memory", [])}

def retrieve_context(state: GraphState) -> GraphState:
    docs = retriever.get_relevant_documents(state["steps"])
    context_text = "\n".join([doc.page_content for doc in docs])
    return {
        "question": state["question"],
        "steps": state["steps"],
        "context": context_text,
        "memory": state.get("memory", [])
    }

def answer_question(state: GraphState) -> GraphState:
    response = safe_llm_invoke(
        answer_prompt.format(context=state["context"], question=state["question"])
    )
    memory = state.get("memory", [])
    memory.append({"question": state["question"], "answer": response})
    return {
        "question": state["question"],
        "steps": state["steps"],
        "context": state["context"],
        "answer": response,
        "memory": memory
    }

# ========= Step 9: Build Graph =========
workflow = StateGraph(GraphState)
workflow.add_node("Decompose", decompose_question)
workflow.add_node("Retrieve", retrieve_context)
workflow.add_node("Answer", answer_question)

workflow.set_entry_point("Decompose")
workflow.add_edge("Decompose", "Retrieve")
workflow.add_edge("Retrieve", "Answer")
workflow.add_edge("Answer", END)

app = workflow.compile()

# ========= Step 10: Run App =========
if __name__ == "__main__":
    try:
        memory: List[Dict[str, str]] = []
        while True:
            user_question = input("Ask your complex MCQ or reasoning question (or type 'exit'):\n> ")
            if user_question.lower() == "exit":
                break
            result = app.invoke({"question": user_question, "memory": memory})
            memory = result["memory"]  # Keep updated memory between turns

            print("\n\U0001f9e0 Final Answer:\n", result["answer"])
            print("\n📃 Memory Log:")
            for i, qa in enumerate(memory, 1):
                print(f"{i}. Q: {qa['question']}\n   A: {qa['answer']}")
    except Exception as e:
        print("\u274c Error:", str(e))




# answer_prompt = PromptTemplate.from_template(
#     "You must answer only using the following context:\n\n{context}\n\nIf the answer is not in the context, say 'I don't know.'\n\nQuestion:\n{question}"
# )