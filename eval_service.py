import asyncio
import itertools
import json
import os
import openai




class State:
    def __init__(self, state, emotion):
        self.state = state
        self.emotion = emotion



class EvalService:
    def __init__(self, api="openai", model_id = "gpt-3.5-turbo"):
        self._api = api
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._model_id = model_id


    async def invoke_llm_async(self, messages, cancel_event=None):
        delay = 0.1

        while True:
            try:
                response = await openai.ChatCompletion.acreate(
                    model=self._model_id,
                    messages=messages,
                    temperature=1.0,  # use 0 for debugging/more deterministic results
                    stream=False
                )
                return response.choices[0].message

            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2

            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2

            except openai.error.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2

            except Exception as e:
                print(f"OpenAI API unknown error: {e}")
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2

    def extract_messages_as_prompt(self, messages):
        prompt = ""
        for message in messages:
            prompt += f"{message}\n"
        return prompt

    async def query_goal_async(self, state:str, messages:[str]):
        messages = messages.copy()
        prompt = self.extract_messages_as_prompt(messages)
        messages = []
        system_prompt = f"""
You are to evaluate the assistent's goal given their current state.

* role: system = this is the system's prompt for the assistant
* role: assistant = text spoke by the assistant
* role: user = text spoken by the user

respond using json format. for example:
{{"goal": "make the user laugh"}}
{{"goal": "spread love and quirky humor"}}
{{"goal": "get attention from the user"}}

the assistants current state is:{state}

"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        responce = await self.invoke_llm_async(messages)
        return responce.content
    
    async def query_state_async(self, messages):
        messages = messages.copy()
        prompt = self.extract_messages_as_prompt(messages)
        messages = []
        system_prompt = f"""
You are to sumerise the assistent's state given the history of the conversation.

* role: system = this is the system's prompt for the assistant
* role: assistant = text spoke by the assistant
* role: user = text spoken by the user

respond using json format. for example:
{{
    "state": "waiting for the user to say something",
    "emotion": "happy",
}}


"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        responce = await self.invoke_llm_async(messages)
        return responce.content