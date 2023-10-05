import asyncio
import itertools
import json
import os
import openai
from pydantic import BaseModel, Field
from typing import Any, Callable, Optional, Sequence

from langchain_helper import convert_pydantic_to_openai_function, create_instance_from_response

class SetState(BaseModel):
    """Function to capture the current state of the assistant. 
    The state will form tha observations in a partially observable MDP.
    The state should be from the perspective of the assistant and include the assistant's percived understanding of the user"""

    assistant_emotion: str = Field(..., description="Current emotion of the assistant")
    user_emotion: str = Field(..., description="Current emotion of the user from the assistant's perspective")
    summary: str = Field(..., description="The summary should capture the state of the assitant and their percived understanding of the user")
    conversation_key_beats: str = Field(..., description="Short summary of the key beats of the conversation. This should be from the perspective of the assistant and should just summarise assitant and user conversation")
    assistant_goal: str = Field(..., description="The assistant's goal")
    user_goal: str = Field(..., description="The assistant's perspective of the user's goal")



class EvalService:
    def __init__(self, api="openai", model_id = "gpt-3.5-turbo"):
        self._api = api
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._model_id = model_id


    async def invoke_llm_async(self, messages, functions, cancel_event=None):
        delay = 0.1
        openai_functions = [convert_pydantic_to_openai_function(f) for f in functions]
        fn_names = [oai_fn["name"] for oai_fn in openai_functions]


        while True:
            try:
                response = await openai.ChatCompletion.acreate(
                    model=self._model_id,
                    messages=messages,
                    temperature=0.0,
                    functions=openai_functions,
                    function_call="auto",
                    stream=False
                )
                output = response.choices[0].message
                function_instance = create_instance_from_response(output, functions)
                return function_instance

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
You are a world class algorithm for defining the current state. 
The input is the conversation history.

* role: system = this is the system's prompt for the assistant
* role: assistant = text spoke by the assistant
* role: user = text spoken by the user

Tip: Make sure to answer in the correct format  
"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        functions = [
            SetState
        ]
        
        responce = await self.invoke_llm_async(messages, functions)
        return responce