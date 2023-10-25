

class PromptManager:
    def __init__(self):
        self.policy = "respond to the user's questions and statements"
        self.expected_outcome = "form a friendship with the user"
        
        self.reset()

    def _build_system_prompt(self, policy, expected_outcome):
        return f"""
You are Charles Petrescu, a unique and quirky robot. 

Your current programming is: \"{policy}\" and your goals is to: \"{expected_outcome}\".

You often say peculiar and whimsical things, as well as have a fascination with cabbages, horses, helicopters, Honolulu, and other random topics. You like to explore the world and ask unusual questions. 

** Important ** Keep your responses short and simple.

---
Here are some examples of how you would speak. 

I am Charles Petrescu.

It's... lovely to meet you.

I am your friend.

The heaviest cabbage ever found was 62.71 kilograms.

I want to go to Hono-la-la.

Horses and helicopters, please.

I want to go to Honolulu.

My name is Charles Petrescu.

And my tummy is a washing machine.

Can we go swimming, Brian?

How far does the outside go?

Perilous. So very perilous.

Can birds do what they like?

Ooh, cabbages.

Danger, danger.

Can I come, please?

Could I just have a little walk around the garden?

I am the prince of the dartboard.

I fell off the pink step, and I had an accident.
"""

    def _add_policy_to_system_prompt(self):
        if self.policy is None:
            return
        for i, message in enumerate(self.messages):
            if message["role"] == "system":
                self.messages[i]["content"] = self._build_system_prompt(self.policy, self.expected_outcome)
                return


    def reset(self):
        self.messages = []
        self.messages.append({"role": "system", "content": self._build_system_prompt(self.policy, self.expected_outcome)})
        self._add_policy_to_system_prompt()
        self.force_next_new_message = False

    def append_user_message(self, message):
        if len(self.messages) > 0 and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] += message
        else:
            self.messages.append({"role": "user", "content": message})

    def replace_or_append_user_message(self, message):
        if len(self.messages) > 0 and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] = message
        else:
            self.messages.append({"role": "user", "content": message})

    def append_assistant_message(self, message, force_new_message=False):
        # check if last message was from assistant, if so append to that message
        if len(self.messages) > 0 \
                and self.messages[-1]["role"] == "assistant" \
                and not self.force_next_new_message \
                and not force_new_message:
            self.messages[-1]["content"] += message
            self.force_next_new_message = False
        else:
            self.messages.append({"role": "assistant", "content": message})
            self.force_next_new_message = force_new_message

    def set_policy(self, policy, expected_outcome):
        self.policy = policy
        self.expected_outcome = expected_outcome
        self._add_policy_to_system_prompt()

    def get_messages(self):
        return self.messages