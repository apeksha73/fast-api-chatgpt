from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from openai import OpenAI
import tempfile
import config
import os

api_key = config.api_key

app = FastAPI()

# Initialize OpenAI Client
client = OpenAI(api_key=api_key)

# Create assistant and vector store at startup
@app.on_event("startup")
async def startup_event():
    global assistant, vector_store, thread

    # Create vector store
    vector_store = client.vector_stores.create(name="Stories")

    # Create assistant
    assistant = client.beta.assistants.create(
        name="Story",
        description="Answers based on uploaded story",
        model="gpt-4o",
        tools=[{"type": "file_search"}]
    )

    # Link vector store to assistant
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
    )

    # Create thread
    thread = client.beta.threads.create()

@app.post("/upload-story/")
async def upload_story(file: UploadFile = File(...)):
        # Make sure the file has the correct suffix
    # Make sure we extract and keep the original extension
    _, ext = os.path.splitext(file.filename)
    if ext.lower() != ".txt":
        return {"error": "Only .txt files are supported for upload."}
    contents = await file.read()
    # Save with .txt suffix to make OpenAI happy
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Upload file to OpenAI
    with open(tmp_path, "rb") as f:
        client.files.create(file=f, purpose="assistants")

        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[f]
        )

    return {"status": file_batch.status, "file_counts": file_batch.file_counts}


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask/")
async def ask_question(data: QuestionRequest):
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=data.question
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        answer = messages.data[0].content[0].text.value
        return {"answer": answer}
    else:
        return {"status": run.status, "message": "Run not completed"}


@app.get("/")
def read_root():
    return {"message": "ChatGPT Assistant API is running!"}
