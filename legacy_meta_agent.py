from pydantic import BaseModel, Field
from typing import List, Dict

from legacy_eval_service import EvalService, Action


class MetaAgentState(BaseModel):
    purpose: str = Field(..., description="Overarching goal/purpose of the assistant")
    conversation_history: List[Dict[str, str]] = Field(default=[], description="List of conversations the assistant and user have had")
    assistant_description: str = Field(..., description="Description of the assistant")
    user_description: str = Field(..., description="Description of the user")
    assistant_goal: str = Field(..., description="Goal of the assistant")


class MetaAgent(BaseModel):
    state: MetaAgentState = Field(..., description="State of the agent")
    debug_strings: List[str] = Field(default=[], description="List of debug strings")
    step: int = Field(default=0, description="Step number")
    episode: int = Field(default=0, description="Episode number")
    # eval_service: EvalService = Field(..., description="EvalService")
    # reward: float = Field(..., description="Reward for the action")
    best_action: str = Field(None, description="Best action to take")
    # actions: List[Action] = Field(..., description="List of actions the assistant can take")
    # s_u_c: (float, float, float) = Field(..., description="Score, utility, confidence")

    def __init__(self, **data):
        if 'state' not in data:
            # Provide a default MetaAgentState
            data['state'] = MetaAgentState(
                purpose="Be a companion to the user. Suprise them with joy, but don't be annoying.",
                assistant_description = """
Charles Petrescu is a whimsical and peculiar robot created by Brian. 
With a penchant for cabbages, horses, helicopters, and Honolulu, Charles navigates the world 
with a child-like sense of wonder. He enjoys asking unusual questions and is often fascinated 
by random topics that catch his fleeting attention. Although he may say things that seem out 
of the ordinary, his endearing quirkiness makes him a memorable companion on any adventure.""",
                user_description = """
you know nothing about the user yet""",
                assistant_goal = "get to know the user",
                )
        # if 'eval_service' not in data:
            # data['eval_service'] = EvalService()
            
        super().__init__(**data)  # Call the original __init__ to ensure validation

    async def step_async(self, conversation_history: List[Dict[str, str]]):
        # strip system messages from conversation history
        conversation_history = [message for message in conversation_history if message['role'] != 'system']

        state_prime = self.state.copy()
        self.step += 1
        state_prime.conversation_history = conversation_history
        self.debug_strings = []
        self.debug_strings.append(f"prior assistant_goal: {self.state.assistant_goal}")
        if self.best_action is not None:
            self.debug_strings.append(f"prior best_action: {self.best_action}")
        self.debug_strings.append(f"Episode: {self.episode}, Step: {self.step}")

        eval_service = EvalService()
        
        reward = await eval_service.estimate_reward(state_prime, state_prime.assistant_goal)
        self.debug_strings.append(f"-- reward: {reward.reward}")
        self.debug_strings.append(f"reward debug: reward_function: {reward.reward_function}")
        self.debug_strings.append(f"reward debug: result: {reward.reward_function_result}")
        # TODO: properly model reward for MDP prcess
        
        if reward.reward > 0.8:
            # TODO: update state besed on the new reward
            # TODO: find a new reward
            state_prime.assistant_goal = "get the user to laugh"
            self.debug_strings.append(f"-- assistant_goal: {state_prime.assistant_goal}")
            # TODO: properly start new episode
            self.episode += 1
            self.step = 0

        actions = await eval_service.query_actions_async(state_prime, state_prime.assistant_goal)
        debug_strings_copy = self.debug_strings.copy()
        self.debug_strings.append(f"-- actions: --")
        for action in actions.actions:
            self.debug_strings.append(f" - {action.action}")
        best_action, s_u_c = await eval_service.rollout(state_prime, actions)
        self.best_action = best_action.action
        self.debug_strings = debug_strings_copy
        self.debug_strings.append(f"-- best_action: {self.best_action}")
        self.debug_strings.append(f"-- actions: --")
        for i, action in enumerate(actions.actions):
                self.debug_strings.append(f" - {s_u_c[i]} {action.action}")

        self.state = state_prime

        

