from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import uuid4

app = FastAPI()

# Allow all CORS headers and origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# In-memory database to store threads and messages
chat_threads: Dict[str, Dict] = {}

# Models
class StartThreadResponse(BaseModel):
    thread_id: str

class Message(BaseModel):
    user: str
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    thread_id: str
    messages: List[Dict[str, str]]

class UserThreadsResponse(BaseModel):
    user_id: str
    threads: List[str]

@app.get("/start_thread", response_model=StartThreadResponse)
async def start_thread(user_id: str):
    # Generate a new unique thread ID
    thread_id = str(uuid4())
    chat_threads[thread_id] = {"user_id": user_id, "messages": []}  # Initialize with an empty message list
    return StartThreadResponse(thread_id=thread_id)

@app.post("/send_message", response_model=ChatResponse)
async def send_message(msg: Message):
    # Check if the thread ID exists
    print(msg)
    if msg.thread_id not in chat_threads:
        raise HTTPException(status_code=404, detail="Thread ID not found")
    
    # Append the message to the thread
    chat_threads[msg.thread_id]["messages"].append({"user": 'user', "message": msg.message})
    chat_threads[msg.thread_id]["messages"].append({"user": 'ai', "message": msg.message}) 
    # Return the chat history for the given thread
    return ChatResponse(thread_id=msg.thread_id, messages=chat_threads[msg.thread_id]["messages"])

# Endpoint to retrieve all messages in a thread for a specific user
@app.get("/get_thread/{thread_id}", response_model=ChatResponse)
async def get_thread(thread_id: str, user_id: str):
    # Check if the thread ID exists and belongs to the user
    if thread_id not in chat_threads or chat_threads[thread_id]["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Thread ID not found or does not belong to the user")
    return ChatResponse(thread_id=thread_id, messages=chat_threads[thread_id]["messages"])

# Endpoint to retrieve all threads for a specific user
@app.get("/get_user_threads/{user_id}", response_model=UserThreadsResponse)
async def get_user_threads(user_id: str):
    # Find all threads belonging to the specified user
    user_threads = [thread_id for thread_id, data in chat_threads.items() if data["user_id"] == user_id]
    return UserThreadsResponse(user_id=user_id, threads=user_threads)


@app.get("/get_messages/{thread_id}", response_model=ChatResponse)
async def get_messages(thread_id: str):
    # Check if the thread ID exists in the database
    if thread_id in chat_threads:
        # Return the chat history for the given thread
        return ChatResponse(thread_id=thread_id, messages=chat_threads[thread_id]["messages"])
    else:
        # If the thread ID is not found, return an empty list of messages
        raise HTTPException(status_code=404, detail="Thread ID not found")