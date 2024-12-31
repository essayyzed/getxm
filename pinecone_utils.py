"""
Utility functions for Pinecone operations shared across services.
"""

import pinecone
import logging
import asyncio
from typing import Optional
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class PineconeUtils:
    @staticmethod
    async def ensure_index_exists(
        index_name: str,
        pinecone_client: pinecone.Pinecone,
        embeddings: Optional[OpenAIEmbeddings] = None
    ) -> bool:
        """
        Check if index exists, create it if it doesn't.
        
        Args:
            index_name: Name of the index to check/create
            pinecone_client: Initialized Pinecone client
            embeddings: OpenAI embeddings instance (only needed for index creation)
            
        Returns:
            bool: True if index exists or was created, False if creation failed
            
        Raises:
            ValueError: If embeddings not provided when index needs to be created
        """
        try:
            # Check if index exists
            if index_name in pinecone_client.list_indexes().names():
                logger.info(f"Index '{index_name}' already exists")
                return True
                
            # If we need to create index, we need embeddings
            if not embeddings:
                raise ValueError("Embeddings required to create new index")
                
            logger.info(f"Creating new Pinecone index: {index_name}")
            # Get embedding dimension by testing
            test_embedding = await embeddings.aembed_query("test")
            dimension = len(test_embedding)
            
            # Create the index
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=pinecone.ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Updated region for free tier
                )
            )
            
            # Wait for index to be ready
            while not pinecone_client.describe_index(index_name).status['ready']:
                await asyncio.sleep(1)
                
            logger.info(f"Successfully created index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error managing Pinecone index: {str(e)}")
            raise
