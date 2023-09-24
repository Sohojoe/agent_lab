import asyncio
import itertools
import json
import os
import openai

from prompt_manager import PromptManager

class FunctionResponse:
    def __init__(self, function_name):
        self.function_name = function_name
        self.arguments_str = ""
        self.arguments_json = ""
        self.done = False

    def append_argument(self, argument):
        self.arguments_str += argument
        try:
            self.arguments_json = json.loads(self.arguments_str)
            self.done = True
        except json.JSONDecodeError:
            pass
    
    def is_done(self):
        return self.done


class ChatService:
    def __init__(self, api="openai", model_id = "gpt-3.5-turbo"):
        self._api = api
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._model_id = model_id

    def _should_we_send_to_voice(self, sentence):
        sentence_termination_characters = [".", "?", "!"]
        close_brackets = ['"', ')', ']']

        temination_charicter_present = any(c in sentence for c in sentence_termination_characters)
 
        # early exit if we don't have a termination character
        if not temination_charicter_present:
            return None

        # early exit the last char is a termination character
        if sentence[-1] in sentence_termination_characters:
            return None
        
        # early exit the last char is a close bracket
        if sentence[-1] in close_brackets:
            return None
        
        termination_indices = [sentence.rfind(char) for char in sentence_termination_characters]
        # Filter out termination indices that are not followed by whitespace or end of string
        termination_indices = [i for i in termination_indices if sentence[i+1].isspace()]
        last_termination_index = max(termination_indices)
        # handle case of close bracket
        while last_termination_index+1 < len(sentence) and sentence[last_termination_index+1] in close_brackets:
            last_termination_index += 1

        text_to_speak = sentence[:last_termination_index+1]
        return text_to_speak
    
    def ignore_sentence(self, text_to_speak):
        # exit if empty, white space or an single breaket
        if text_to_speak.isspace():
            return True
        # exit if not letters or numbers
        has_letters = any(char.isalpha() for char in text_to_speak)
        has_numbers = any(char.isdigit() for char in text_to_speak)
        if not has_letters and not has_numbers:
            return True
        return False

    async def get_responses_as_sentances_async(self, prompt_manager:[PromptManager], cancel_event=None):
        llm_response = ""
        current_sentence = ""
        delay = 0.1

        while True:
            try:
                response = await openai.ChatCompletion.acreate(
                    model=self._model_id,
                    messages=prompt_manager.messages,
                    functions=prompt_manager.functions,
                    function_call=prompt_manager.function_call,
                    temperature=prompt_manager.temperature,
                    stream=True
                )

                function_response = None
                async for chunk in response:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    chunk_message = chunk['choices'][0]['delta']
                    function_call = chunk_message.get('function_call')
                    if function_call:
                        if (function_call.get('name')):
                            function_response = FunctionResponse(function_call.get('name'))
                        if (function_call.get('arguments')):
                            function_response.append_argument(function_call.get('arguments'))
                        yield function_response, function_response.is_done()

                    elif chunk_message.get('content'):
                        chunk_text = chunk_message['content']
                        current_sentence += chunk_text
                        llm_response += chunk_text
                        text_to_speak = self._should_we_send_to_voice(current_sentence)
                        if text_to_speak:
                            current_sentence = current_sentence[len(text_to_speak):]
                            yield text_to_speak, True
                        else:
                            yield current_sentence, False

                if cancel_event is not None and cancel_event.is_set():
                    return
                if len(current_sentence) > 0:
                    yield current_sentence, True
                return

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