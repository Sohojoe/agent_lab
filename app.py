import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn


chat_history = []
debug_info = []
user_typing_feed = ""
steps = 0


app = FastAPI()

async def emit_debug():
    global steps
    global debug_info
    debug_info = []
    debug_info.append(f"---- debug info ----")
    debug_info.append(f"typing_in_progress: {user_typing_feed}")
    debug_info.append(f"steps: {steps}")
    await sio.emit("update_debug", debug_info)

async def main_loop():
    global steps
    global debug_info
    while True:
        # Your main loop code here
        steps += 1

        await emit_debug()
        
        # Sleep for 1/30th of a second to achieve 30 FPS
        await asyncio.sleep(1 / 30)

# Start the main loop when the application starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(main_loop())

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
sio_asgi_app = socketio.ASGIApp(socketio_server=sio, other_asgi_app=app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@sio.event
async def connect(sid, environ):
    print(f"User connected: {sid}")

@sio.event
async def typing_in_progress(sid, data):
    global user_typing_feed
    user_typing_feed=data

@sio.event
async def complete_sentence(sid, data):
    global user_typing_feed
    user_typing_feed = ""
    chat_history.append(data)
    await sio.emit("update_chat", chat_history)


if __name__ == "__main__":
    uvicorn.run("app:sio_asgi_app", host="0.0.0.0", port=8000, reload=True)