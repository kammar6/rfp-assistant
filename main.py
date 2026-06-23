from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import fitz  # library to read pdfs
import httpx # we need this to make async calls to our local ollama server
import uuid  # using this to generate unique ids for the database entries
import hashlib

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

class SearchQuery(BaseModel):
    question: str
    limit: int = 3 # default to returning the top 3 matches

class AskQuery(BaseModel):
    question: str
    limit: int = 3
    model: str = "qwen3.5:9b"

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

async def generate_answer(context_chunks: list[str], question: str, model: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        f"You are an expert assistant helping answer questions about RFP documents.\n"
        f"Use ONLY the excerpts below to answer the question. "
        f"If the answer is not contained in the excerpts, say so clearly.\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["response"]

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
            vector = await get_embedding(chunk)
            
            # create a deterministic ID based on the text content
            # if we upload the exact same text again, it generates the same ID and overwrites
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            deterministic_id = str(uuid.UUID(chunk_hash))
            
            # create a qdrant data point
            point = PointStruct(
                id=deterministic_id, 
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
    
@app.post("/search")
async def search_documents(query: SearchQuery):
    try:
        # step 1: turn the user's question into a math vector using ollama
        question_vector = await get_embedding(query.question)

        # step 2: ask qdrant to find the chunks closest to our question vector
        # using the new query_points function since search() got removed in the new update
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=question_vector, 
            limit=query.limit
        ).points # we have to grab the .points list from the response object

        # step 3: clean up the output so it is easy to read
        matches = []
        for hit in search_results:
            matches.append({
                "confidence_score": hit.score, # higher is better (closer to 1.0)
                "text": hit.payload["text"],
                "source": hit.payload["source_file"]
            })

        return {
            "status": "success",
            "question": query.question, 
            "matches": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/ask")
async def ask_question(query: AskQuery):
    try:
        question_vector = await get_embedding(query.question)

        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=question_vector,
            limit=query.limit
        ).points

        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        context_chunks = [hit.payload["text"] for hit in search_results]
        sources = list({hit.payload["source_file"] for hit in search_results})

        answer = await generate_answer(context_chunks, query.question, query.model)

        return {
            "status": "success",
            "question": query.question,
            "answer": answer,
            "sources": sources,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

@app.get("/documents")
def list_documents():
    try:
        # tally up how many chunks belong to each source file
        counts: dict[str, int] = {}
        next_offset = None

        # scroll through the whole collection in batches until there is nothing left
        while True:
            points, next_offset = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False  # we only need the metadata, not the heavy vectors
            )

            for point in points:
                source = point.payload.get("source_file", "unknown")
                counts[source] = counts.get(source, 0) + 1

            # when qdrant has no more pages it returns a null offset
            if next_offset is None:
                break

        documents = [
            {"source_file": source, "chunk_count": count}
            for source, count in counts.items()
        ]

        return {
            "status": "success",
            "total_documents": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")