import asyncio
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import socketio
import uvicorn
from meta_agent import MetaAgent

from prompt_manager import PromptManager
from respond_to_prompt import RespondToPromptAsync
from response_state_manager import ResponseStateManager
from sensory_stream import SensoryStream


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
        self.chat_history = ["lazy init"]
        self.debug_info = []
        self.user_typing_feed = ""
        self.response_state_manager = ResponseStateManager()
        self.prompt_manager = PromptManager()
        self.output_history = []
        self.respond_to_prompt = None
        self.respond_to_prompt_task = None
        self.meta_agent = MetaAgent()
        self.sensory_stream = SensoryStream()
        self.sensory_stream.append_event("An unknown user entered...")

        @sio.event
        async def connect(sid, environ):
            print(f"User connected: {sid}")

        @sio.event
        async def typing_in_progress(sid, data):
            self.user_typing_feed = data

        @sio.event
        async def complete_sentence(sid, prompt):
            self.user_typing_feed = ""
            response_preview_text = self.response_state_manager.pretty_print_current_responses()
            if len(response_preview_text) > 0:
                self.add_output_to_history(response_preview_text)
            self.add_output_to_history(f"👨 {prompt}\n")
            self.prompt_manager.replace_or_append_user_message(prompt)
            self.sensory_stream.append_user_message(prompt)
            self.respond_to_prompt = RespondToPromptAsync(self.response_state_manager)
            self.respond_to_prompt_task = asyncio.create_task(self.respond_to_prompt.run(prompt, self.prompt_manager.messages))
            response_step_obs, response_state = self.response_state_manager.reset_episode()

    def add_output_to_history(self, output):
        self.output_history.append(output)
        if len(self.output_history) > 10:
            self.output_history.pop(0)


    async def emit_debug(self):
        self.debug_info = []
        self.debug_info.append(f"---- debug info ----")
        self.debug_info.append(f"episode: {self.response_state_manager.episode}")
        self.debug_info.append(f"step: {self.response_state_manager.step}")
        task_status = "n/a"
        if self.respond_to_prompt_task is not None:
            if self.respond_to_prompt_task.done():
                task_status = "done"
            else:
                task_status = "running"
        self.debug_info.append(f"respond_to_prompt_task: {task_status}")

        self.debug_info.append(f"---- MetaAgent debug info ----")
        for debug_string in self.meta_agent.debug_strings:
            self.debug_info.append(debug_string)
        await sio.emit("update_debug", self.debug_info)

    async def emit_chat_history(self, human_preview_text):
        list_of_strings = self.output_history.copy()
        robot_preview_text = self.response_state_manager.pretty_print_preview_text()
        response_preview_text = self.response_state_manager.pretty_print_current_responses()
        if len(robot_preview_text) > 0:
            response_preview_text += robot_preview_text+"  \n"
        list_of_strings.append(response_preview_text)
        if len(human_preview_text) > 0:
            list_of_strings.append(human_preview_text)
        if len(list_of_strings) > 10:
            list_of_strings.pop(0)
        chat_history = []
        for item in list_of_strings:
            lines = item.split('\n')
            for line in lines:
                if len(line) > 0:
                    chat_history.append(line)
        if len(chat_history) == 0:
            chat_history = ["...waiting..."]
        if chat_history != self.chat_history:
            await sio.emit("update_chat", chat_history)
        self.chat_history = chat_history

    async def eval_loop(self):
        while True:
            try:
                await self.meta_agent.step_async(self.sensory_stream)

                await asyncio.gather(
                    asyncio.sleep(10)
                )    
            except Exception as e:
                # trace the exception
                print(f"Exception in eval_loop: {e}")
                trace = traceback.format_exc()
                print(f"trace: {trace}")
                await asyncio.sleep(10)

    async def main_loop(self):
        prior_meta_agent_policy = None
        while True:
            response_step_obs, response_state = self.response_state_manager.begin_next_step()
            should_review_meta_agent = True
            prompt = self.user_typing_feed
            human_preview_text = ""
            if len(prompt):
                human_preview_text = f"👨❓ {prompt}"
                should_review_meta_agent = False

            for new_response in response_step_obs.llm_responses:
                self.prompt_manager.append_assistant_message(new_response)
                self.sensory_stream.append_assistant_message(new_response)
                should_review_meta_agent = False

            if self.respond_to_prompt_task is not None and not self.respond_to_prompt_task.done():
                should_review_meta_agent = False

            if should_review_meta_agent:
                if self.meta_agent.current_policy is not None and self.meta_agent.current_policy != prior_meta_agent_policy:
                    prior_meta_agent_policy = self.meta_agent.current_policy
                    self.prompt_manager.set_policy(self.meta_agent.current_policy.policy, self.meta_agent.current_policy.expected_outcome)
                    response_preview_text = self.response_state_manager.pretty_print_current_responses()
                    if len(response_preview_text) > 0:
                        self.add_output_to_history(response_preview_text)
                    # self.add_output_to_history(f"🧠 {prompt}")

                    self.respond_to_prompt = RespondToPromptAsync(self.response_state_manager)
                    self.respond_to_prompt_task = asyncio.create_task(self.respond_to_prompt.run(prompt, self.prompt_manager.messages))
                    response_step_obs, response_state = self.response_state_manager.reset_episode()

            await asyncio.gather(
                self.emit_chat_history(human_preview_text),
                self.emit_debug(),
                asyncio.sleep(1 / 30)
            )            


@app.on_event("startup")
async def startup_event():
    main = Main()
    asyncio.create_task(main.main_loop())
    asyncio.create_task(main.eval_loop())

if __name__ == "__main__":
    uvicorn.run("app:sio_asgi_app", host="0.0.0.0", port=8000, reload=True)


