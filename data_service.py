from fastapi import FastAPI, HTTPException
import os
import requests
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import logging
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
import nltk
import asyncio
import pinecone
from pathlib import Path
from functools import lru_cache
from contextlib import asynccontextmanager
from pinecone_utils import PineconeUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to capture all levels
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add a file handler for DEBUG level
debug_handler = logging.FileHandler('data_service_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(debug_handler)

class EnvironmentManager:
    """Manages environment variables and configuration."""
    
    @staticmethod
    @lru_cache()
    def load_environment() -> None:
        """Load environment variables with caching."""
        load_dotenv(find_dotenv(), override=True)
        logger.info("Environment variables loaded")
    
    @staticmethod
    def get_required_env(key: str) -> str:
        """Get a required environment variable or raise an error."""
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

class ProcessDataRequest(BaseModel):
    """Request model for data processing endpoint."""
    file_path: str = Field(..., description="Path to data file")
    index_name: str = Field(..., description="Name of the Pinecone index to use")
    chunk_size: int = Field(default=1000, ge=100, le=2000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Overlap between chunks")

class ProcessDataResponse(BaseModel):
    """
    Response model for data processing endpoint.
    
    Attributes:
        status: Success or error status
    """
    status: str = Field(..., description="Status of the operation")

class PineconeManager:
    """Manages Pinecone initialization and operations."""
    
    def __init__(self, index_name: str = None):
        self.api_key = EnvironmentManager.get_required_env("PINECONE_API_KEY")
        self.environment = EnvironmentManager.get_required_env("PINECONE_ENVIRONMENT")
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "default-index")
        self.pc = None
        self.index = None
        self.vector_store = None
        
    async def initialize(self, embeddings: OpenAIEmbeddings) -> None:
        """Initialize Pinecone connection and index."""
        try:
            self.pc = pinecone.Pinecone(api_key=self.api_key)
            
            # Use shared utility to ensure index exists
            await PineconeUtils.ensure_index_exists(
                index_name=self.index_name,
                pinecone_client=self.pc,
                embeddings=embeddings
            )
            
            self.index = self.pc.Index(self.index_name)
            self.vector_store = PineconeVectorStore(
                index=self.index,
                embedding=embeddings
            )
            logger.info(f"Successfully initialized Pinecone with index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise

class DataProcessor:
    """
    Handles document processing, chunking, and storage in vector database.
    
    This class manages the entire pipeline of processing documents:
    1. Loading documents from files
    2. Splitting documents into optimal chunks
    3. Converting chunks to embeddings
    4. Storing embeddings in vector database
    
    Attributes:
        vector_store (PineconeVectorStore): Instance of Pinecone vector store for document storage
        text_splitter (RecursiveCharacterTextSplitter): Text splitter for chunking documents
    
    Example:
        ```python
        processor = DataProcessor(vector_store)
        result = await processor.process_file(
            file_path="documents/text.txt",
            chunk_size=1000,
            chunk_overlap=200
        )
        ```
    """
    
    def __init__(self, vector_store: PineconeVectorStore):
        """
        Initialize the DataProcessor.
        
        Args:
            vector_store: Instance of PineconeVectorStore for document storage
        """
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter()
    
    async def process_file(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> None:
        """Process a file and store its chunks in the vector store."""
        try:
            logger.info(f"Starting to process file: {file_path}")
            logger.debug(f"Using chunk size: {chunk_size}, overlap: {chunk_overlap}")
            
            # Load documents
            documents = await self._load_documents(Path(file_path))
            logger.debug(f"Loaded {len(documents)} documents")
            
            # Split documents
            split_texts = await self._split_documents(
                documents,
                chunk_size,
                chunk_overlap
            )
            logger.debug(f"Split into {len(split_texts)} chunks")
            
            # Store documents
            await self._store_documents(split_texts)
            logger.info("File processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
    
    async def _load_documents(self, path: Path) -> List[Document]:
        """
        Load documents from a file.
        
        This method handles the initial document loading phase:
        1. Reads the file from disk
        2. Converts file content to Document objects
        3. Handles asynchronous file operations
        
        Args:
            path: Path object pointing to the file to load
            
        Returns:
            List of Document objects containing the file content
            
        Raises:
            IOError: If file reading fails
            Exception: For other loading errors
            
        Note:
            Uses asyncio to handle file I/O operations asynchronously
            to prevent blocking the event loop.
        """
        loader = TextLoader(str(path))
        return await asyncio.get_running_loop().run_in_executor(
            None,
            loader.load
        )
    
    async def _split_documents(
        self,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Split documents into smaller chunks for optimal processing.
        
        This method handles the document chunking phase:
        1. Configures the text splitter with chunk parameters
        2. Splits documents into smaller chunks
        3. Processes chunks to ensure consistent format
        
        The chunking process is crucial for:
        - Optimal embedding generation
        - Improved retrieval accuracy
        - Context preservation
        
        Args:
            documents: List of Document objects to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks ready for embedding
            
        Example:
            Original text: "The quick brown fox jumps over the lazy dog"
            With chunk_size=20 and chunk_overlap=5:
            -> ["The quick brown", "brown fox jumps", "jumps over the", "the lazy dog"]
            
        Note:
            Chunk overlap helps maintain context between chunks and
            improves the quality of retrieval results.
        """
        self.text_splitter.chunk_size = chunk_size
        self.text_splitter.chunk_overlap = chunk_overlap
        split_texts = self.text_splitter.split_documents(documents)
        return [
            text.page_content if isinstance(text, Document) else text
            for text in split_texts
        ]
    
    async def _store_documents(self, texts: List[str]) -> None:
        """
        Store document chunks in the vector database.
        
        This method handles the final storage phase:
        1. Converts text chunks to Document objects
        2. Generates unique IDs for each chunk
        3. Stores documents with their embeddings in Pinecone
        
        The storage process:
        1. Each chunk is converted to a Document object
        2. Each Document gets a UUID for unique identification
        3. Documents are embedded and stored in Pinecone
        4. Embeddings are indexed for efficient retrieval
        
        Args:
            texts: List of text chunks to store
            
        Raises:
            Exception: If storage in vector database fails
            
        Note:
            - UUIDs ensure each chunk can be uniquely identified
            - Pinecone handles the embedding generation internally
            - Storage is optimized for quick retrieval
        """
        documents = [Document(page_content=text) for text in texts]
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        logger.info("Documents added to vector store")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Handles initialization and cleanup of application resources:
    1. NLTK data download
    2. Pinecone initialization
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None
        
    Raises:
        Exception: If initialization fails
    """
    try:
        # Initialize NLTK
        nltk.download(['punkt', 'punkt_tab'])
        logger.info("NLTK data downloaded successfully")
        
        yield
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        # Cleanup code (if needed)
        logger.info("Shutting down application")

# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="Data Processing Service",
    lifespan=lifespan
)

# Initialize services
EnvironmentManager.load_environment()

@app.post("/process", response_model=ProcessDataResponse)
async def process_data(request: ProcessDataRequest) -> ProcessDataResponse:
    """Process data endpoint."""
    try:
        # Initialize Pinecone with specified index
        pinecone_manager = PineconeManager(index_name=request.index_name)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=EnvironmentManager.get_required_env("OPENAI_API_KEY")
        )
        await pinecone_manager.initialize(embeddings)
        
        # Initialize data processor
        processor = DataProcessor(pinecone_manager.vector_store)
        
        # Process the file
        await processor.process_file(
            file_path=request.file_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        return ProcessDataResponse(status="success")
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
