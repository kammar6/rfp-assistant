from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Connect to the local Qdrant instance running in Docker
qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "rfp_documents"

# Modern FastAPI lifespan context manager replacing on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup execution: Ensure the vector collection exists
    collections = qdrant_client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            # nomic-embed-text outputs vectors with 768 dimensions
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    yield
    # Shutdown execution would go here if needed

# Initialize FastAPI with the lifespan handler
app = FastAPI(title="RFP Assistant API", lifespan=lifespan)

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "API is running", "message": "Ready to process RFPs"}

@app.post("/process-text")
def process_text(input_data: TextInput):
    # Subsequent implementation will route text to Ollama and store the resulting vector
    return {
        "status": "success",
        "received_text": input_data.text,
        "message": "Text received by Python Backend. Qdrant collection is ready."
    }