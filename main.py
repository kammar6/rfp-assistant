from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RFP Assistant API")

# A simple data model defining the expected input
class TextInput(BaseModel):
    text: str

# A test endpoint to check if the API is reachable
@app.get("/")
def read_root():
    return {"status": "API is running", "message": "Ready to process RFPs"}

# The endpoint where n8n will send the data later
@app.post("/process-text")
def process_text(input_data: TextInput):
    # Logic for the vector database and AI will be added here later
    return {
        "status": "success",
        "received_text": input_data.text,
        "message": "Text received by Python Backend"
    }