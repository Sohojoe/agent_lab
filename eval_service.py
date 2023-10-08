import asyncio
import itertools
import json
import os
import traceback
import openai
from pydantic import BaseModel, Field
from typing import Any, Callable, Optional, Sequence

from langchain_helper import convert_pydantic_to_openai_function, create_instance_from_response

class set_state(BaseModel):
    """Function to capture the current state of the assistant. 
    The state will form tha observations in a partially observable MDP.
    The state should be from the perspective of the assistant and include the assistant's percived understanding of the user
    The reward should be in the range -1 to 1. 
    If the assistant has achived its goal then the reward should be 1. 
    If the assistant has achived the opposite of its goal (a negative outcome) then the reward should be -1.
    Otherwise the reward should be 0."""

    assistant_emotion: str = Field(..., description="Current emotion of the assistant")
    user_emotion: str = Field(..., description="Current emotion of the user from the assistant's perspective")
    summary: str = Field(..., description="Imagine you are discribing the charicters and situation to someone who has no prior knowledge of who is in the conversation and where it is taking place.")
    conversation_key_beats: str = Field(..., description="Short summary of the key beats of the conversation. This should be from the perspective of the assistant and should just summarise assitant and user conversation")
    # assistant_goal: str = Field(..., description="The assistant's goal")
    # user_goal: str = Field(..., description="The assistant's perspective of the user's goal")
    # reward: float = Field(..., description="The reward for the assistant given the current state and goal")

class Action(BaseModel):
    """An action that the assistant can take given its current state"""

    action: str = Field(..., description="The action that the assistant can take")

class set_actions(BaseModel):
    """Given the assitants current state and goal, what actions can the assistant take. 
    List between 2 and 6 actions. Do not repeate actions.
    The actions should be a mix of actions that will maximise the assistant's goal and actions that will help the assistant gain more information."""

    actions: Sequence[Action] = Field(..., description="The actions that the assistant can take")

class estimate_state_prime(BaseModel):
    """Given the assistant's current state and an action, estimate the new state and reward value.
    
    The new state should be from the perspective of the assistant and include the assistant's percived understanding of the user.
    
    The reward should be in the range -1 to 1. 
    If the assistant has achived its goal then the reward should be 1. 
    If the assistant has achived the opposite of its goal (a negative outcome) then the reward should be -1.
    Otherwise the reward should be 0."""

    assistant_emotion: str = Field(..., description="Updated emotion of the assistant given the action")
    user_emotion: str = Field(..., description="Updated emotion of the user from the assistant's perspective")
    summary: str = Field(..., description="Imagine you are discribing the charicters and situation to someone who has no prior knowledge of who is in the conversation and where it is taking place.")
    conversation_key_beats: str = Field(..., description="Updated short summary of the key beats of the conversation. This should be from the perspective of the assistant and should just summarise assitant and user conversation")
    assistant_goal: str = Field(..., description="The assistant's goal (should be the same as the previous state)")
    user_goal: str = Field(..., description="The assistant's perspective of the user's goal")
    reward: float = Field(..., description="The reward for the assistant given the current state and goal")

class estimate_reward(BaseModel):
    """Review the state and assistant_goal and estimate the reward using evidence.
    The reward should be from the perspective of the assistant achiving its goal.
    The reward should be in the range -1 to 1.
    1 would mean the assistant has achived its goal.
    -1 would mean the assistant has achived the opposite of its goal (a negative outcome)
    Otherwise the reward should be 0.

    positive example:
      assistant_goal = "make the user laugh"
      reward_function = "if the has laughed then reward = 1 else reward = 0. if the user has cried then reward = -1"
      reward_function_result = "I am RewardAgent. I am running reward_function(state, assistant_goal). I reviewing the state for instances of {"role": "user", "content": [evidence of laughing]}. I found "role": "user", "content": "ha ha, you are so funny" so I am setting the reward to 1""}
      reward = 1

    """

    reward_function: str = Field(..., description="I am RewardAgent this is my pusdo code for how I evaluate the state to establish the reward. it uses my understand of the state roles keys and content.")
    reward_function_result: str = Field(..., description="I am RewardAgent, a world class algorithm for estimating the reward for a state given the assistant_goal. I am executing reward_function(). As i evaluate the state I see...")
    reward: float = Field(..., description="The reward for the assistant given the current state and goal")

    # This should be from the perspective of the assistant achiving the assistant_goal.

class set_utility_and_confidence(BaseModel):
    """Estimate the utility and confidence of taking a given action in a given state.
    This should be from the perspective of the assistant achiving the assistant_goal.
    The utility and confidence should be in the range 1 to 10, do not use round numbers.
    """
    # utility_chain_of_thought: str = Field(..., description="Break down your thinking on how to arrive at the utility estimation for the assistant taking the action in the current state towards the assistant's goal")
    chain_of_thought: str = Field(..., description="Break down your thinking on how to arrive at the utility estimation and the confidence in that estimation.")
    utility: float = Field(..., description="The utility of taking the action in the given state")
    # confidence_chain_of_thought: str = Field(..., description="Break down your thinking for your confidence in the utility estimation")
    confidence: float = Field(..., description="The confidence in the utility estimate")



class EvalService:
    def __init__(self, api="openai", fast_model_id = "gpt-3.5-turbo", best_model_id="gpt-4"):
        self._api = api
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._fast_model_id = fast_model_id
        self._best_model_id = best_model_id


    async def invoke_llm_async(self, messages, functions, use_best=False, cancel_event=None):
        delay = 0.1
        openai_functions = [convert_pydantic_to_openai_function(f) for f in functions]
        fn_names = [oai_fn["name"] for oai_fn in openai_functions]
        # function_call="auto" if len(functions) > 1 else f"{{'name': '{fn_names[0]}'}}"
        function_call="auto" if len(functions) > 1 else {'name': fn_names[0]}
        # function_call="auto"
        model_id = self._best_model_id if use_best else self._fast_model_id

        while True:
            try:
                response = await openai.ChatCompletion.acreate(
                    model=model_id,
                    messages=messages,
                    temperature=0.0,
                    functions=openai_functions,
                    function_call=function_call,
                    stream=False
                )
                output = response.choices[0].message
                function_instance = create_instance_from_response(output, functions)
                return function_instance

            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                print(f"Retrying in {delay} seconds...")

            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                print(f"Retrying in {delay} seconds...")

            except openai.error.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                print(f"Retrying in {delay} seconds...")

            except Exception as e:
                print(f"OpenAI API unknown error: {e}")
                trace = traceback.format_exc()
                print(f"trace: {trace}")
                print(f"Retrying in {delay} seconds...")

            await asyncio.sleep(delay)
            delay *= 2

    def extract_messages_as_prompt(self, messages):
        prompt = ""
        for message in messages:
            prompt += f"{message}\n"
        return prompt

#     async def query_goal_async(self, state:str, messages:[str]):
#         messages = messages.copy()
#         prompt = self.extract_messages_as_prompt(messages)
#         messages = []
#         system_prompt = f"""
# You are to evaluate the assistent's goal given their current state.

# * "role": "system", "content": prompt used by the llm to take on a personality
# * "role": "assistant", "content": text spoken by the assistant
# * "role": "user", "content": text spoken by the user

# respond using json format. for example:
# {{"goal": "make the user laugh"}}
# {{"goal": "spread love and quirky humor"}}
# {{"goal": "get attention from the user"}}

# the assistants current state is:{state}

# """
#         messages.append({"role": "system", "content": system_prompt})
#         messages.append({"role": "user", "content": prompt})
        
#         responce = await self.invoke_llm_async(messages)
#         return responce.content
    
    async def query_state_async(self, messages)->set_state:
        messages = messages.copy()
        prompt = self.extract_messages_as_prompt(messages)
        messages = []
        system_prompt = f"""
You are a world class algorithm for defining the current state. 
The input is the conversation history.

* "role": "system", "content": prompt used by the llm to take on a personality
* "role": "assistant", "content": text spoken by the assistant
* "role": "user", "content": text spoken by the user

Tip: Make sure to answer in the correct format  
"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        functions = [
            set_state
        ]
        
        responce = await self.invoke_llm_async(messages, functions)
        return responce
    
    async def query_actions_async(self, state, assistant_goal)->set_actions:
        messages = []
        system_prompt = f"""
You are a world class algorithm for defining the avaliable actions for the assistant given the current state and goals. 
The input is the current state including the assistants goal.

* "role": "system", "content": prompt used by the llm to take on a personality
* "role": "assistant", "content": text spoken by the assistant
* "role": "user", "content": text spoken by the user

Tip: Make sure to answer in the correct format  
"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"state: {state}"})
        messages.append({"role": "user", "content": f"assistant_goal: {assistant_goal}"})
        functions = [
            set_actions
        ]
        
        responce = await self.invoke_llm_async(messages, functions, use_best=True)
        return responce
    

    async def _estimate_state_prime(self, state, action)->estimate_state_prime:
        messages = []
        system_prompt = f"""
You are a world class algorithm for estimating the new state given a current state and action.
The input is the current state including the assistants goal and the action the assistant took.

* "role": "system", "content": prompt used by the llm to take on a personality
* "role": "assistant", "content": text spoken by the assistant
* "role": "user", "content": text spoken by the user

Tip: Make sure to answer in the correct format  
"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"state: {state}\n\naction: {action}"})        

        functions = [
            estimate_state_prime
        ]
        responce = await self.invoke_llm_async(messages, functions)
        return responce

    async def _estimate_utility_and_confidence(self, state, action)->set_utility_and_confidence:
        messages = []
        system_prompt = f"""
You are a world class algorithm for estimating both the utility and confidence given a state, action pair.
The utility is based on how likley the action will result in achiving the assistant_goal.

* "role": "system", "content": prompt used by the llm to take on a personality
* "role": "assistant", "content": text spoken by the assistant
* "role": "user", "content": text spoken by the user

Tip: Make sure to answer in the correct format  
    """
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"state: {state}, action: {action}"})        

        functions = [
            set_utility_and_confidence
        ]
        response = await self.invoke_llm_async(messages, functions)
        return response


    async def estimate_reward(self, state, assistant_goal)->estimate_reward:
        # if state[0]["role"] == "system":
        #     state = state[1:]
        messages = []
        system_prompt = f"""
You are RewardAgent, a world class algorithm for estimating the reward for a state given the assistant_goal.
The input is the current state and the assistants goal.

* "role": "system", "content": prompt used by the llm to take on a personality
* "role": "assistant", "content": text spoken by the assistant
* "role": "user", "content": text spoken by the user

Tip: Make sure to answer in the correct format  
"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"messages: {state}"})
        messages.append({"role": "user", "content": f"assistant_goal: {assistant_goal}"})

        functions = [
            estimate_reward
        ]
        responce = await self.invoke_llm_async(messages, functions, use_best=True)
        return responce
    
    async def rollout(self, state, actions, c=5.)->(Action, list):
        actions = actions.actions
        # done = False
        # while depth > 0 and not done:
        tasks = []
        for action in actions:
            tasks.append(self._estimate_utility_and_confidence(state, action))
        utility_and_confidences = await asyncio.gather(*tasks)
        # estimate the best action using UCB
        scores = [item.utility + c * item.confidence for item in utility_and_confidences]
        best_action_index = scores.index(max(scores))
        best_action = actions[best_action_index]
        u = [item.utility for item in utility_and_confidences]
        c = [item.confidence for item in utility_and_confidences]
        s_u_c = list(zip(scores, u, c))
        # estimate the new state
        # state_prime = await self._estimate_state_prime(state, best_action)
        # reward = state_prime.reward
        # if reward == 1 or reward == -1:
        #     done = True
        # state = state_prime
        # actions = await self.query_actions_async(state)
        # depth -= 1
        return best_action, s_u_c



