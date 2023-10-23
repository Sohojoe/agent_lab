from pydantic import ConfigDict, BaseModel, Field
from typing import List, Dict
from active_inference_service import ActiveInferenceService, Policy, PolicyIsCompleteEnum, select_policy_fn

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
            data['generative_model'] = GenerativeModelFactory(empty=True)
        # if 'eval_service' not in data:
            # data['eval_service'] = EvalService()
            
        super().__init__(**data)  # Call the original __init__ to ensure validation

    async def step_async(self, sensor_stream: SensoryStream):
        # strip system messages from conversation history
        pritty_sensor_stream = sensor_stream.pritty_print()

        self.step += 1
        self.debug_strings = []
        self.debug_strings.append(f"--- observations ---")
        observations = self.generative_model.get_observations()
        for observation in observations:
            self.debug_strings.append(f" - {observation.document}")
        self.debug_strings.append(f"--- sensor_stream ---")
        for line in pritty_sensor_stream.split("\n"):
            self.debug_strings.append(f" - {line}")

        active_inference_service = ActiveInferenceService()

        if self.current_policy is not None:
            self.debug_strings.append(f"--- current policy ---")
            self.debug_strings.append(f"- {self.current_policy.policy}")
            self.debug_strings.append(f"--- updating model...")
            model_updates = await active_inference_service.update_generative_model(sensor_stream, self.generative_model, self.current_policy)
            for belief in model_updates.add_beliefs:
                self.debug_strings.append(f" - add: {belief.belief}")
            for belief in model_updates.delete_beliefs:
                self.debug_strings.append(f" - delete: {belief.belief}")
            for belief in model_updates.edit_beliefs:
                self.debug_strings.append(f" - update: {belief.old_belief} -> {belief.new_belief}")
            self.debug_strings.append(f" - {model_updates.policy_is_complete}")
            if model_updates.policy_is_complete == PolicyIsCompleteEnum.complete or \
                    model_updates.policy_is_complete == PolicyIsCompleteEnum.interrupt_policy:
                self.current_policy = None

        if self.current_policy is None:
            self.debug_strings.append(f"--- selecting new policy...")
            select_policies_result:select_policy_fn = await active_inference_service.select_policy(sensor_stream, self.generative_model)
            self.current_policy = select_policies_result.policies[select_policies_result.selected_policy_idx]
            self.debug_strings.append(f"-- selected policy -")
            self.debug_strings.append(f"- {self.current_policy.policy}")
            self.debug_strings.append(f"-- free energy causes -")
            for free_energy in select_policies_result.free_energy_causes:
                self.debug_strings.append(f"- {free_energy.cause} ({free_energy.estimated_free_energy})")
            self.debug_strings.append(f"-- policies -")
            for policy in select_policies_result.policies:
                self.debug_strings.append(f"- policy: {policy.policy}")
                self.debug_strings.append(f"  expected_outcome: {policy.expected_outcome}")
                self.debug_strings.append(f"  estimated_free_energy_reduction: {policy.estimated_free_energy_reduction}")
                self.debug_strings.append(f"  probability_of_success: {policy.probability_of_success}")
                self.debug_strings.append(f" {policy.estimated_free_energy_reduction * policy.probability_of_success}")

        

        

