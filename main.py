from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import fitz  # library to read pdfs
import httpx # we need this to make async calls to our local ollama server
import uuid  # using this to generate unique ids for the database entries

# connect to the qdrant container
qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "rfp_documents"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # check if our table/collection exists when the server boots up
    collections = qdrant_client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            # nomic-embed-text outputs 768 numbers per vector
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    yield

app = FastAPI(title="RFP Assistant API", lifespan=lifespan)

class TextInput(BaseModel):
    text: str

# helper function to split text into overlapping chunks
# the overlap prevents us from cutting a sentence in half
def get_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # shift the start index, but keep some of the old text (overlap)
        start += chunk_size - overlap
        
    return chunks

# function to ask ollama to turn our text into numbers (embeddings)
async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        # calling the local ollama background service we set up earlier
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=60.0 # giving it 60 seconds because the first run can be slow
        )
        response.raise_for_status() # crash if something goes wrong
        
        # parse the json and grab the embedding array
        return response.json()["embedding"]

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # quick check to make sure the user actually uploaded a pdf
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        # load the file into ram
        file_content = await file.read()
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        extracted_text = ""

        # loop through all the pages and combine the text
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            extracted_text += page.get_text()

        total_pages = pdf_document.page_count
        pdf_document.close() # clean up memory

        # split the big text string into chunks
        document_chunks = get_text_chunks(extracted_text)
        
        points = []
        
        # go through every chunk and get its vector from ollama
        for chunk in document_chunks:
            # this might take a few seconds depending on the gpu
            vector = await get_embedding(chunk)
            
            # create a qdrant data point with a random id, the vector, and the text payload
            point = PointStruct(
                id=str(uuid.uuid4()), 
                vector=vector,
                payload={"text": chunk, "source_file": file.filename}
            )
            points.append(point)
            
        # upload all the points to qdrant in one batch
        if points:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

        # return a summary to n8n so we know it worked
        return {
            "status": "success",
            "filename": file.filename,
            "total_pages": total_pages,
            "total_chunks_vectorized_and_saved": len(document_chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")