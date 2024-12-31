"""
Chatbot Service Module

This module provides a FastAPI-based chatbot service that uses:
- OpenAI's GPT model for natural language processing
- Pinecone for vector storage and retrieval
- LangChain for orchestrating the QA pipeline

The service provides a conversational interface that can answer
questions based on previously processed documents.
"""

from fastapi import FastAPI, HTTPException
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
from pathlib import Path
import re
import pinecone
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pinecone_utils import PineconeUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to capture all levels
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add a file handler for DEBUG level
debug_handler = logging.FileHandler('debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(debug_handler)

class EnvironmentManager:
    """Manages environment variables and configuration."""
    
    @staticmethod
    def load_environment(env_path: Path) -> None:
        """
        Load environment variables from .env file using python-dotenv.
        
        Args:
            env_path: Path to .env file
            
        Raises:
            FileNotFoundError: If .env file doesn't exist
            ValueError: If environment variables are invalid
        """
        try:
            # Check if file exists
            if not env_path.exists():
                raise FileNotFoundError(f".env file not found at {env_path}")
            
            logger.debug(f"Loading .env from: {env_path.absolute()}")
            
            # Load using python-dotenv
            result = load_dotenv(env_path, override=True)
            if not result:
                raise ValueError(f"Failed to load .env file from {env_path}")
            
            # Verify key variables
            openai_key = os.environ.get("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            # Log success but don't show the actual key
            logger.info("Environment variables loaded successfully")
            logger.debug(f"OpenAI API key loaded (length: {len(openai_key)}, prefix: {openai_key[:7]}...)")
            
        except Exception as e:
            logger.error(f"Error loading environment: {str(e)}")
            raise

class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.
    
    Attributes:
        question: The user's question to be answered
        index_name: The name of the Pinecone index to query
    """
    question: str = Field(..., min_length=1, max_length=1000)
    index_name: str = Field(..., description="Name of the Pinecone index to query")

class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.
    
    Attributes:
        status: Success or error status
        answer: The answer to the user's question
    """
    status: str
    answer: str

class CleanTextOutputParser(BaseOutputParser[str]):
    """
    Parser that cleans and formats model output.
    
    Removes unnecessary whitespace, newlines, and formats
    the text for consistent output.
    """
    
    def parse(self, text: str) -> str:
        """
        Parse and clean the input text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and formatted text
        """
        return ' '.join(text.strip().split())

class ChatbotService:
    """Main service class for chatbot functionality.
    
    Handles initialization of all required components:
    - LLM (Language Model)
    - Embeddings
    - Vector Store
    - QA Chain
    """
    
    def __init__(self, index_name: str = None):
        """Initialize the chatbot service components."""
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.output_parser = CleanTextOutputParser()
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "default-index")
        self.pc = None
    
    async def initialize(self):
        """Initialize all required services."""
        try:
            # Initialize in correct order
            await self._init_llm()
            await self._init_embeddings()
            await self._init_vector_store()  # This needs embeddings
            await self._init_qa_chain()      # This needs vector_store
            logger.info("ChatbotService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatbotService: {str(e)}")
            raise

    async def _init_llm(self) -> None:
        """Initialize the Language Model."""
        try:
            logger.debug("Initializing LLM with OpenAI")
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            raise
    
    async def _init_embeddings(self) -> None:
        """Initialize the embeddings model."""
        try:
            logger.debug("Initializing OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Embeddings initialization failed: {str(e)}")
            raise
    
    async def _init_vector_store(self):
        """Initialize the Pinecone vector store."""
        try:
            self.pc = pinecone.Pinecone(
                api_key=os.environ["PINECONE_API_KEY"]
            )
            
            # Check if index exists (don't create if it doesn't)
            await PineconeUtils.ensure_index_exists(
                index_name=self.index_name,
                pinecone_client=self.pc
            )
            
            index = self.pc.Index(self.index_name)
            self.vector_store = PineconeVectorStore(
                index=index,
                embedding=self.embeddings
            )
            logger.info(f"Vector store initialized with index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def _init_qa_chain(self):
        """Initialize the QA chain."""
        try:
            # Define the QA prompt
            template = """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
            qa_prompt = ChatPromptTemplate.from_template(template)
            
            # Create the QA chain with the vector store retriever
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(),
                chain_type_kwargs={"prompt": qa_prompt},
                return_source_documents=True
            )
            logger.info("QA chain initialized successfully")
        except Exception as e:
            logger.error(f"QA chain initialization failed: {str(e)}")
            raise

    async def get_answer(self, question: str) -> Dict[str, str]:
        """
        Get answer for a given question.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing status and answer
        """
        try:
            logger.info(f"Processing question: {question}")
            result = await self.qa_chain.ainvoke({"query": question})
            answer = result["result"] if "result" in result else result["answer"]
            parsed_response = self.output_parser.parse(answer)
            logger.info("Answer generated successfully")
            return {
                "status": "success",
                "answer": parsed_response
            }
        except Exception as e:
            logger.error(f"Failed to get answer: {str(e)}")
            return {
                "status": "error",
                "answer": str(e)
            }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for FastAPI application.
    
    Handles initialization and cleanup of application resources.
    """
    try:
        # Load environment variables first
        env_path = Path(__file__).parent / '.env'
        EnvironmentManager.load_environment(env_path)
        
        yield
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        # Cleanup (if needed)
        logger.info("Shutting down application")

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot Service",
    description="AI-powered chatbot service for document Q&A",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint for question answering.
    """
    try:
        # Initialize chatbot service with specified index
        service = ChatbotService(index_name=request.index_name)
        await service.initialize()
        
        # Get answer
        result = await service.get_answer(request.question)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
