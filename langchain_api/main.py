from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from controller import ChatController

import uvicorn
import os
import uuid

app = FastAPI()
chat_controller = ChatController()

origins = [
    "http://localhost",
    "http://localhost:3000",  # Assuming your React app is running on port 3000
    "https://educai.site",
    "https://www.educai.site",
    "https://delightful-field-0a372d61e.5.azurestaticapps.net",
    "https://demo.aiwolf.dev/",
    "https://www.demo.aiwolf.dev/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Include OPTIONS method
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}



@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message")
    chat_id = data.get("chat_id")
    program_uuid = data.get("programId")  # Capture program_uuid from the request

    if not chat_id:
        chat_id = str(uuid.uuid4())

    try:
        # Pass program_uuid to chat_with_history for program-specific vector store logic
        #response = chat_controller.chat_with_history(chat_id, user_message, program_uuid)
        responses=chat_controller.generate_ai_response(user_message,chat_id)
        return {"chat_id": chat_id, "program_uuid": program_uuid, "reply": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 2020))  # Adjust port as needed
    )
