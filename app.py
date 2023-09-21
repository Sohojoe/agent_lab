from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
app = FastAPI()
sio_asgi_app = socketio.ASGIApp(socketio_server=sio, other_asgi_app=app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_history = []
debug_info = []

@sio.event
async def connect(sid, environ):
    print(f"User connected: {sid}")

@sio.event
async def typing_in_progress(sid, data):
    # debug_info.append(data)
    debug_info = [
        "\n"
        f"typing_in_progress: {data}"
        ]
    await sio.emit("update_debug", debug_info)

@sio.event
async def complete_sentence(sid, data):
    chat_history.append(data)
    await sio.emit("update_chat", chat_history)


if __name__ == "__main__":
    uvicorn.run("app:sio_asgi_app", host="0.0.0.0", port=8000, reload=True)