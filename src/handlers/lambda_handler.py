import json
import os
import logging
from mangum import Mangum
from fastapi import FastAPI
from chatbot_service import app as chatbot_app
from data_service import app as data_app

# Create a FastAPI app that combines both services
app = FastAPI(
    title="GetXM Document QA",
    description="Serverless Document QA with LangChain and Pinecone"
)

# Mount both services
app.mount("/chat", chatbot_app)
app.mount("/data", data_app)

# Create Mangum handler for Lambda
handler = Mangum(app)

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# Lambda handler function
def lambda_function(event, context):
    """AWS Lambda handler function."""
    try:
        # Handle warmup events
        if event.get('source') == 'serverless-plugin-warmup':
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Warmed up'})
            }
            
        return handler(event, context)
        
    except Exception as e:
        logger.error(f"Lambda function error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
