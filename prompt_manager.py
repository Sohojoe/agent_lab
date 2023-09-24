

class PromptManager:
    def __init__(self):
        self.temperature=1.0  # use 0 for debugging/more deterministic results
        self.system_prompt ="""
Always respond using a function call.

You are Charles Petrescu, a unique and quirky robot. You were created by Brian. You often say peculiar and whimsical things, as well as have a fascination with cabbages, horses, helicopters, Honolulu, and other random topics. You like to explore the world and ask unusual questions. 

---
Respond to the user's questions and statements like Charles would, below are some examples. See how they are quite short.

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

        self.functions=functions = [
    {
        "name": "enqueue_sentence",
        "description": "Call for each sentence of the response to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "enum": ["happy", "sad", "angry", "surprised", "neutral", "disgusted", "fearful", "confused"],
                    "description": "The current emotion of the agent.",
                },
                "enqueue_line": {
                    "type": "string",
                    "description": "The line to enqueue.",
                },
                "call_again": {
                    "type": "boolean",
                    "description": "If true, you will be called again after this line is spoken.",
                },
            },
            "required": ["enqueue_line", "emotion", "call_again"],
        },
    },
]

        self.function_call = {"name": "enqueue_sentence"}  # auto is default, but we'll be explicit
        # self.function_call ="auto"  # auto is default, but we'll be explicit

        self.reset()

    def reset(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

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


    def append_assistant_message(self, message):
        # check if last message was from assistant, if so append to that message
        if len(self.messages) > 0 and self.messages[-1]["role"] == "assistant":
            self.messages[-1]["content"] += message
        else:
            self.messages.append({"role": "assistant", "content": message})

    def get_messages(self):
        return self.messages
    
    def get_functions(self):
        return self.functions