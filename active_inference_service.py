import asyncio
from enum import Enum
import itertools
import json
import os
import traceback
import openai
from pydantic import BaseModel, Field
from typing import Any, Callable, ForwardRef, List, Optional, Sequence
from generative_model import GenerativeModel

from langchain_helper import convert_pydantic_to_openai_function, create_instance_from_response
from sensory_stream import SensoryStream

Policy = ForwardRef('Policy')
update_generative_model_fn = ForwardRef('update_generative_model_fn')
select_policy_fn = ForwardRef('select_policy_fn')

class ActiveInferenceService:
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
                    temperature=1.0,
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

    async def select_policy(self, sensory_stream: SensoryStream, generative_model: GenerativeModel)->select_policy_fn:
        messages = []
        system_prompt = f"""
You are an Artificial Intelligence expert specializing in Active Inference, the Free Energy Principle, and the Markov Blanket. Your landmark research showed that Large Language Models (LLMs) like GPT-4 can perform Active Inference. 

* The LLM's prompt and its fixed weights were seen as a generating model: gm = f(prompt, fixed_weights)
* The LLM's attention and its static weights behave as the Markov Blanket.
* You concluded that the fixed LLM weights contain a wealth of hidden states, beliefs, and desires about the world that do not need to be learned.
* Therefore, attention should be given to using the prompt to only model the unique hidden states, beliefs, and desires relevant to the AI assistant's context or those that contradict the fixed weights.

You are now playing an advisor's role to an AI assistant, which defines its hidden states, beliefs, and desires using the User prompt.

Tip: Make sure to answer in the correct format  
"""
        messages.append({"role": "system", "content": system_prompt})
        observations = generative_model.get_observations()
        observations = [o.document for o in observations]
        stream = sensory_stream.pritty_print().split("\n")
        state = {
            "beliefs": observations,
            "sensory_stream": stream,
        }
        state = json.dumps(state)
        messages.append({"role": "user", "content": f"state: {state}"})
        functions = [
            select_policy_fn
        ]
        select_policies_result: select_policy_fn = await self.invoke_llm_async(messages, functions, use_best=True)
        policy_impacts = [p.estimated_free_energy_reduction * p.probability_of_success for p in select_policies_result.policies]
        # find index of max(policy_impacts)
        max_index = policy_impacts.index(max(policy_impacts))
        select_policies_result.selected_policy_idx = max_index
        return select_policies_result


    async def update_generative_model(self, sensory_stream: SensoryStream, generative_model: GenerativeModel, cur_policy: Policy)->update_generative_model_fn:
        messages = []
        system_prompt = f"""
You are an Artificial Intelligence expert specializing in Active Inference, the Free Energy Principle, and the Markov Blanket. Your landmark research showed that Large Language Models (LLMs) like GPT-4 can perform Active Inference. 

* The LLM's prompt and its fixed weights were seen as a generating model: gm = f(prompt, fixed_weights)
* The LLM's attention and its static weights behave as the Markov Blanket.
* You concluded that the fixed LLM weights contain a wealth of hidden states, beliefs, and desires about the world that do not need to be learned.
* Therefore, attention should be given to using the prompt to only model the unique hidden states, beliefs, and desires relevant to the AI assistant's context or those that contradict the fixed weights.

You are now playing an advisor's role to an AI assistant, which defines its hidden states, beliefs, and desires using the System prompt.

Tip: Make sure to answer in the correct format  
"""
        messages.append({"role": "system", "content": system_prompt})
        observations = generative_model.get_observations()
        observations = [o.document for o in observations]
        stream = sensory_stream.pritty_print().split("\n")
        state = {
            "beliefs": observations,
            "sensory_stream": stream,
            "current_policy": cur_policy.model_dump(),
        }
        state = json.dumps(state)
        messages.append({"role": "user", "content": f"state: {state}"})
        functions = [
            update_generative_model_fn
        ]
        updates: update_generative_model_fn = await self.invoke_llm_async(messages, functions, use_best=True)
        for belief in updates.add_beliefs:
            generative_model.add_observation(belief.belief)
        for belief in updates.delete_beliefs:
            generative_model.drop_observation_by_document(belief.belief)
        for belief in updates.edit_beliefs:
            generative_model.edit_observation_by_document(belief.old_belief, belief.new_belief)
        return updates

class Policy(BaseModel):
    """A set of actions the assistant takes to reduce free energy"""
    policy: str = Field(..., description="The set of actions in the policy")
    expected_outcome:str = Field(..., description="expected changes that the policy will produce in perception, hidden_states, and/or beliefs")
    estimated_free_energy_reduction: float = Field(...,  description= "how much free energy the policy will remove from the system. int between 1 and 10")
    probability_of_success: float = Field(...,  description= "how likley the assistant is to succeed with this policy, between 0 and 1")

# class EditBelifActionEnum(str, Enum):
#     create = "create"
#     edit = "edit"
#     delete = "delete"


class AddBelief(BaseModel):
    """Create a belief"""
    belief: str = Field(..., description="something the assistent beleives about themselves, the user, or the world")

class DeleteBelief(BaseModel):
    """Delete a belief"""
    belief: str = Field(None, description="belief to detele. MUST match an existing belief")


class EditBelief(BaseModel):
    """Edit, create, or delete a belief"""
    new_belief: str = Field(..., description="something the assistent beleives about themselves, the user, or the world")
    old_belief: str = Field(None, description="belief to update. MUST match an existing belief")

class PolicyIsCompleteEnum(str, Enum):
    """Is the policy complete? Should the assistant continue or interrupt?"""
    continue_policy = "continue",
    complete = "complete",
    interrupt_policy = "interrupt",

class update_generative_model_fn(BaseModel):
    """Your tasks: 
    #1 evaluate if the assistant should update its hidden_states, beliefs
    #2 evaluate if the policy is complete or should be interrupted and new policies evaluated.
    """

    add_beliefs: Optional[List[AddBelief]] = Field(..., description="only add new beliefs if they are unqiue and do not contradict the fixed weights")
    delete_beliefs: Optional[List[DeleteBelief]] = Field(..., description="delete beliefs that are no longer relevant or are repetative")
    edit_beliefs: Optional[List[EditBelief]] = Field(..., description="update beliefs that have changed")
    policy_is_complete: PolicyIsCompleteEnum = Field(..., description="Determines if the policy is complete or should be interrupted and new policies evaluated.")


class FreeEnergy(BaseModel):
    cause: str = Field(..., description="cause of free energy / uncertanty in the system")
    estimated_free_energy: float = Field(..., description="estimated size of free energy / uncertanty this cases is the system")
    
class select_policy_fn(BaseModel):
    """ Your task: 
    Point out 3-5 causes of free energy / uncertanty in the system to help the assistant select the best policy.
    In your response, propose 3-5 multi step policies that will reduce the maximum free energy in the system.
    Finally, pick out the policy you predict will reduce the maximum free energy (estimated_free_energy_reduction * probability_of_success).
    All aspects should be considered from the assistant's point of view.

    Note: The types of polices the assistant can execute successful are speech based in terms of specific questions, statements, chains of conversation. or to pause and wait for the user.
    Note: the types of expected_outcomes should be specific predictions about changes to the perception stream, the hidden_states, the set of beliefs.
    """

    free_energy_causes: List[FreeEnergy] = Field(..., description="3 to 5 causes of free energy / uncertanty in the system")
    policies: List[Policy] = Field(..., description="3 to 5 multi step policies that will reduce the maximum free energy in the system")
    selected_policy_idx: int = Field(..., description="the index of the policy you predict will reduce the maximum free energy (estimated_free_energy_reduction * probability_of_success)")