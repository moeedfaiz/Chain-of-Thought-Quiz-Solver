from flask import Flask, render_template, request, jsonify
from main import app as langgraph_app, GraphState  # Reuse your LangGraph logic

app = Flask(__name__)
memory_log = []  # Global memory between requests

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global memory_log
    data = request.json
    question = data.get("question")

    result = langgraph_app.invoke(GraphState(question=question, memory=memory_log))
    memory_log = result["memory"]  # Update memory

    return jsonify({
        "answer": result["answer"],
        "memory": memory_log
    })

if __name__ == "__main__":
    app.run(debug=True)
