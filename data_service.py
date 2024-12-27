from fastapi import FastAPI
import os
import requests
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import logging
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import nltk
import asyncio

# Initialize NLTK data
try:
    nltk.download(['punkt', 'punkt_tab'])
except Exception as e:
    logging.error(f"Error downloading NLTK data: {str(e)}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables with override
os.environ.clear()  # Clear existing environment variables
load_dotenv(find_dotenv(), override=True)  # Force reload with override

app = FastAPI()

# Initialize embeddings
embeddings = GPT4AllEmbeddings()

class ProcessDataRequest(BaseModel):
    file_path: str = None  # Optional path, will use DATA_PATH from env if not provided

@app.post("/process")
async def process_data(request: ProcessDataRequest):
    """
    Unified endpoint that handles both data ingestion and embedding creation.
    """
    try:
        # Use provided path or fall back to environment variable
        file_path = request.file_path or os.environ.get("DATA_PATH")
        if not file_path:
            raise ValueError("No file path provided and DATA_PATH environment variable is not set")
            
        logger.info(f"Starting data processing for file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found at {file_path}")
            return {"status": "error", "message": f"File not found at {file_path}"}
            
        # Load and process the text file
        loader = TextLoader(file_path)
        try:
            documents = await asyncio.get_running_loop().run_in_executor(None, loader.load)
            logger.info(f"Successfully loaded {len(documents)} documents")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_texts = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_texts)} chunks")
            
            # Generate embeddings and create FAISS index
            vector_store = FAISS.from_documents(split_texts, embeddings)
            logger.info("Created FAISS index")

            # Ensure the directory exists
            index_path = os.environ.get("FAISS_INDEX_PATH")
            os.makedirs(index_path, exist_ok=True)
            logger.info(f"Saving index to: {index_path}")

            # Save the FAISS index
            vector_store.save_local(index_path)
            logger.info("Successfully saved FAISS index")
            
            return {
                "status": "success", 
                "message": "Data processed and embeddings stored successfully",
                "details": {
                    "documents_loaded": len(documents),
                    "chunks_created": len(split_texts),
                    "index_path": index_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error during document processing: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
            
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
