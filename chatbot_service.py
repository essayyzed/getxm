# chatbot_service.py
from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import Tuple
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

app = FastAPI()

# Define request model
class ChatRequest(BaseModel):
    question: str

gpt4 = ChatOpenAI(model="gpt-4", temperature=0.0)
# Load Llama2 via Ollama
llm = OllamaLLM(
    model="llama3.1:8b",
    temperature=0.0,  # Slightly increased temperature for more natural responses
)

# Load the FAISS index
embeddings = GPT4AllEmbeddings()
vector_store = FAISS.load_local(os.environ["FAISS_INDEX_PATH"], embeddings, allow_dangerous_deserialization=True)

# Custom prompt template for QA
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:""")
])

# Create the chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": qa_prompt},
)

class CleanTextOutputParser(BaseOutputParser[str]):
    """Parser that removes newlines and extra spaces from text."""
    
    def parse(self, text: str) -> str:
        # Clean up multiple spaces and newlines
        return ' '.join(text.strip().split())

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        result = qa_chain.invoke(request.question)
        # Use our custom parser
        parsed_response = CleanTextOutputParser().parse(result["result"])
        return {"status": "success", "answer": parsed_response}
    except Exception as e:
        return {"status": "error", "message": str(e)}