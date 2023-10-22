from pydantic import ConfigDict, BaseModel, Field
from typing import List, Dict
from active_inference_service import ActiveInferenceService, Policy

# from eval_service import EvalService, Action
from generative_model import GenerativeModel, GenerativeModelFactory
from sensory_stream import SensoryStream


class MetaAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    generative_model: GenerativeModel = Field(..., description="State of the agent")
    debug_strings: List[str] = Field(default=[], description="List of debug strings")
    step: int = Field(default=0, description="Step number")
    episode: int = Field(default=0, description="Episode number")
    # eval_service: EvalService = Field(..., description="EvalService")
    # reward: float = Field(..., description="Reward for the action")
    # best_action: str = Field(None, description="Best action to take")
    # actions: List[Action] = Field(..., description="List of actions the assistant can take")
    # s_u_c: (float, float, float) = Field(..., description="Score, utility, confidence")
    current_policy: Policy = Field(None, description="Current policy")

    def __init__(self, **data):
        if 'state' not in data:
            data['generative_model'] = GenerativeModelFactory()
        # if 'eval_service' not in data:
            # data['eval_service'] = EvalService()
            
        super().__init__(**data)  # Call the original __init__ to ensure validation

    async def step_async(self, sensor_stream: SensoryStream):
        # strip system messages from conversation history
        conversation_history = sensor_stream.pritty_print()

        self.step += 1
        self.debug_strings = []
        self.debug_strings.append(f"conversation_history")
        for line in conversation_history.split("\n"):
            self.debug_strings.append(f" - {line}")

        active_inference_service = ActiveInferenceService()

        self.debug_strings.append(f"--- update model ---")
        model_updates = await active_inference_service.update_generative_model(sensor_stream, self.generative_model, self.current_policy)
        for belief_update in model_updates.modify_beliefs:
            self.debug_strings.append(f"belief_update: {belief_update}")
        self.debug_strings.append(f"policy_is_complete: {model_updates.policy_is_complete}")
        

        

