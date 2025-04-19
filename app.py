from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
# Deprecation warning: HuggingFaceEmbeddings and HuggingFaceHub in this import may be deprecated; consider updating if needed.
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

if not PINECONE_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Missing API keys. Set PINECONE_API_KEY and HUGGINGFACE_API_KEY in your .env file.")

# Set environment variables for Pinecone & HuggingFaceHub
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Initialize Flask
app = Flask(__name__)

# Load Embeddings using HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to an existing Pinecone index
index_name = "medibot1"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the LLM from HuggingFaceHub: using Mistral-Nemo-Instruct-2407 here
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    model_kwargs={"temperature": 0.4, "max_new_tokens": 500},
    task="text-generation"
)

# Define system prompt with context placeholder
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Avoid repeating information. If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
).strip()

# Create prompt template from messages
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create a chain to combine the documents into a final answer
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Deduplication logic to remove duplicate text in retrieved documents
def remove_duplicate_docs(docs):
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs

# Define a custom retrieval chain that deduplicates context
class DedupedRetrievalChain:
    def __init__(self, retriever, combine_docs_chain):
        self.retriever = retriever
        self.combine_docs_chain = combine_docs_chain

    def invoke(self, inputs):
        docs = self.retriever.get_relevant_documents(inputs["input"])
        unique_docs = remove_duplicate_docs(docs)
        # The combine chain expects context in a specific structure.
        return self.combine_docs_chain.invoke({"input": inputs["input"], "context": unique_docs})

# Create the final RAG chain
rag_chain = DedupedRetrievalChain(retriever, question_answer_chain)

# Flask Routes
@app.route("/")
def index():
    return render_template("chat.html")  # Ensure a chat.html exists in the templates folder.

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    # Invoke the RAG chain; note that the returned value might be a string.
    response_val = rag_chain.invoke({"input": msg})
    # Adjust for response type: if it's a dict, use the "answer" key; otherwise assume it's a string.
    if isinstance(response_val, dict):
        answer = response_val.get("answer", "")
    else:
        answer = response_val
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
