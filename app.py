import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn


app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
sio_asgi_app = socketio.ASGIApp(socketio_server=sio, other_asgi_app=app)

app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

class Main:
    def __init__(self):
        self.chat_history = []
        self.debug_info = []
        self.user_typing_feed = ""
        self.steps = 0

        @sio.event
        async def connect(sid, environ):
            print(f"User connected: {sid}")

        @sio.event
        async def typing_in_progress(sid, data):
            self.user_typing_feed = data

        @sio.event
        async def complete_sentence(sid, data):
            self.user_typing_feed = ""
            lines = data.split('\n')
            for line in lines:
                self.chat_history.append(line)
            await sio.emit("update_chat", self.chat_history)

    async def emit_debug(self):
        self.debug_info = []
        self.debug_info.append(f"---- debug info ----")
        self.debug_info.append(f"typing_in_progress: {self.user_typing_feed}")
        self.debug_info.append(f"steps: {self.steps}")
        await sio.emit("update_debug", self.debug_info)

    async def main_loop(self):
        while True:
            self.steps += 1
            await self.emit_debug()
            await asyncio.sleep(1 / 30)


@app.on_event("startup")
async def startup_event():
    main = Main()
    asyncio.create_task(main.main_loop())

if __name__ == "__main__":
    uvicorn.run("app:sio_asgi_app", host="0.0.0.0", port=8000, reload=True)


