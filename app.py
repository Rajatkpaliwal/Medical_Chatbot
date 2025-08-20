from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from src.prompt import *
from src.history import get_chat_history, add_to_history, clear_history
import os
load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = download_embeddings()

index_name = "medical-chatbot"
              
docSearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding=embeddings
)

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

base_retriever = docSearch.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})

chatModel = ChatGroq(model = "Llama3-8b-8192")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chatModel, base_retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

@app.route("/")
def index():
    clear_history()
    return render_template('chat.html')

@app.route("/get", methods = ["POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"Input: {input}")

    current_chat_history = get_chat_history()

    print(input)
    response = rag_chain.invoke({
        "input": input,
        "chat_history": current_chat_history.messages
    })

    add_to_history(input, response["answer"])

    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080, debug = True)