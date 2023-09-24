
from asyncio import Queue, TaskGroup
import asyncio
import json
import time
from agent_response import AgentResponse
from chat_service import ChatService, FunctionResponse
from prompt_manager import PromptManager
from response_state_manager import ResponseStateManager


class RespondToPromptAsync:
    def __init__(self, response_state_manager:ResponseStateManager):
        self.response_state_manager = response_state_manager

    async def prompt_to_llm(self, prompt:str, prompt_manager:[PromptManager]):
        chat_service = ChatService()
        self.speed_limit = None
        # self.speed_limit = 1/7.
        self.last_time = time.time()

        async def respect_speed_limit():
            if self.speed_limit is not None and self.speed_limit > 0:
                to_wait = self.speed_limit - (time.time() - self.last_time)
                if to_wait > 0:
                    await asyncio.sleep(to_wait)
            self.last_time = time.time()

        def procress_text_response(agent_response:AgentResponse, text:str, is_complete_sentance:bool):
            if chat_service.ignore_sentence(text):
                is_complete_sentance = False
            if not is_complete_sentance:
                agent_response['llm_preview'] = text
                self.response_state_manager.set_llm_preview(text)
                return
            agent_response['llm_preview'] = ''
            agent_response['llm_sentence'] = text
            agent_response['llm_sentences'].append(text)
            self.response_state_manager.add_llm_response_and_clear_llm_preview(text)
            print(f"{agent_response['llm_sentence']} id: {agent_response['llm_sentence_id']} from prompt: {agent_response['prompt']}")
            sentence_response = agent_response.make_copy()
            # TODO add any chains on sentence here
            agent_response['llm_sentence_id'] += 1

        def procress_enqueue_sentence(agent_response:AgentResponse, function:FunctionResponse, is_done:bool):
            if is_done:
                text = function.arguments_json.get('enqueue_line')
                emotion = function.arguments_json.get('emotion')
                agent_response['llm_preview'] = ''
                agent_response['llm_sentence'] = text
                agent_response['llm_sentences'].append(text)
                self.response_state_manager.add_llm_response_and_clear_llm_preview(text)
                print(f"{agent_response['llm_sentence']} id: {agent_response['llm_sentence_id']} from prompt: {agent_response['prompt']}")
                sentence_response = agent_response.make_copy()
                # TODO add any chains on sentence here
                agent_response['llm_sentence_id'] += 1
                return
            part_json = None
            try:
                part_json = json.loads(f"{function.arguments_str}}}")
            except json.JSONDecodeError:
                try:
                    part_json = json.loads(f"{function.arguments_str}\"}}")
                except json.JSONDecodeError:
                    pass
            if not part_json:
                return
            text = part_json.get("enqueue_line")
            emotion = part_json.get("emotion")
            if text:
                agent_response['llm_preview'] = text
                self.response_state_manager.set_llm_preview(text)
            

        async with TaskGroup() as tg:
            agent_response = AgentResponse(prompt)
            async for chunk, is_done in chat_service.get_responses_as_sentances_async(prompt_manager):
                await respect_speed_limit()
                if type(chunk) is str:
                    procress_text_response(agent_response, chunk, is_done)
                elif type(chunk) is FunctionResponse and chunk.function_name == "enqueue_sentence":
                    procress_enqueue_sentence(agent_response, chunk, is_done)
                elif type(chunk) is FunctionResponse:
                    # create not supported exception
                    raise Exception(f"unsupported function: {chunk.function_name}")
                else:
                    raise Exception(f"unsupported type: {type(chunk)}")

    async def run(self, prompt:str, prompt_manager:[PromptManager]):
        self.task_group_tasks = []
        async with TaskGroup() as tg:  # Use asyncio's built-in TaskGroup
            t1 = tg.create_task(self.prompt_to_llm(prompt, prompt_manager))
            self.task_group_tasks.extend([t1])

    async def terminate(self):
        # Cancel tasks
        all_tasks = []
        if self.task_group_tasks:
            for task in self.task_group_tasks:
                task.cancel()
            all_tasks.extend(self.task_group_tasks)
        await asyncio.gather(*all_tasks, return_exceptions=True)
        self.task_group_tasks = []
