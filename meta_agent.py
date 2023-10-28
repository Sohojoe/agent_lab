import asyncio
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
            generative_model = GenerativeModelFactory()
            data['generative_model'] = generative_model
            
        super().__init__(**data)  # Call the original __init__ to ensure validation

    async def step_async(self, sensor_stream: SensoryStream):
        # strip system messages from conversation history
        pritty_sensor_stream = sensor_stream.pritty_print()

        self.step += 1
        self.debug_strings = []
        self.debug_strings.append(f"--- generative_model ---")
        dump = self.generative_model.model_dump()
        for key in self.generative_model.model_fields:
            self.debug_strings.append (f"-- {key}")
            for item in dump[key]:
                self.debug_strings.append (f" - {item}")
        self.debug_strings.append(f"--- sensor_stream ---")
        for line in pritty_sensor_stream.split("\n"):
            self.debug_strings.append(f" - {line}")

        active_inference_service = ActiveInferenceService()

        if self.current_policy is not None:
            self.debug_strings.append(f"--- current policy ---")
            self.debug_strings.append(f"- {self.current_policy.policy}")
            self.debug_strings.append(f"--- updating model...")

            model_task = asyncio.create_task(active_inference_service.update_generative_model(sensor_stream, self.generative_model, self.current_policy))
            policy_task = asyncio.create_task(active_inference_service.track_policy_progress(sensor_stream, self.generative_model, self.current_policy))
            model_update, policy_update = await asyncio.gather(model_task, policy_task)

            self.debug_strings.append(f"--- generative_model changes ---")
            old_generatilve_model = self.generative_model
            self.generative_model = model_update.generative_model
            if (len(self.generative_model.assistant_beliefs- old_generatilve_model.assistant_beliefs)):
                self.debug_strings.append (f"-- assistant_beliefs")
                for item in self.generative_model.assistant_beliefs- old_generatilve_model.assistant_beliefs:
                    self.debug_strings.append (f" - {item}")
            if (len(self.generative_model.assistant_desires- old_generatilve_model.assistant_desires)):
                self.debug_strings.append (f"-- assistant_desires")
                for item in self.generative_model.assistant_desires- old_generatilve_model.assistant_desires:
                    self.debug_strings.append (f" - {item}")
            if (len(self.generative_model.uncertainty_in_the_system- old_generatilve_model.uncertainty_in_the_system)):
                self.debug_strings.append (f"-- uncertainty_in_the_system")
                for item in self.generative_model.uncertainty_in_the_system- old_generatilve_model.uncertainty_in_the_system:
                    self.debug_strings.append (f" - {item}")

            # self.debug_strings.append(f" policy_progress: {policy_update.policy_progress}")
            self.debug_strings.append(f" progress: {policy_update.question_1}")
            self.debug_strings.append(f" outcome achieved?: {policy_update.question_2}")
            self.debug_strings.append(f" still likley?: {policy_update.question_3}")
            self.debug_strings.append(f" - {policy_update.policy_is_complete.name}")

            if policy_update.policy_is_complete == PolicyIsCompleteEnum.complete or \
                    policy_update.policy_is_complete == PolicyIsCompleteEnum.interrupt_policy:
                self.current_policy = None

        if self.current_policy is None:
            self.debug_strings.append(f"--- selecting new policy...")
            select_policies_result:select_policy_fn = await active_inference_service.select_policy(sensor_stream, self.generative_model)
            self.current_policy = select_policies_result.policies[select_policies_result.selected_policy_idx]
            self.debug_strings.append(f"-- selected policy -")
            self.debug_strings.append(f"- {self.current_policy.policy}")
            # self.debug_strings.append(f"-- free energy causes -")
            # for free_energy in select_policies_result.free_energy_causes:
            #     self.debug_strings.append(f"- {free_energy.cause} ({free_energy.estimated_free_energy})")
            self.debug_strings.append(f"-- policies -")
            for policy in select_policies_result.policies:
                self.debug_strings.append(f"- policy: {policy.policy}")
                self.debug_strings.append(f"  expected_outcome: {policy.expected_outcome}")
                self.debug_strings.append(f"  estimated_free_energy_reduction: {policy.estimated_free_energy_reduction}")
                self.debug_strings.append(f"  probability_of_success: {policy.probability_of_success}")
                self.debug_strings.append(f" {policy.estimated_free_energy_reduction * policy.probability_of_success}")

        

        

