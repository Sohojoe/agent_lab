from datetime import datetime

class ResponseStepObservations:
    def __init__(self, episode, step):
        self.timestamp = datetime.utcnow()
        self.episode = episode
        self.step = step
        self.llm_preview = ''
        self.llm_responses = []
        self.tts_raw_chunk_ids = [] 

    def __str__(self):
        state = ', '.join(f'{k}={v}' for k, v in self.__dict__.items() if k not in {'episode', 'step', 'timestamp'})
        return f'episode={self.episode}, step={self.step}, timestamp={self.timestamp}, \nstate=({state})'

class ResponseState:
    def __init__(self, episode, step):
        self.timestamp = datetime.utcnow()
        self.episode = episode
        self.step = step
        self.current_responses = []
        self.speech_chunks_per_response = []
        self.llm_preview = ''

    def __str__(self):
        state = ', '.join(f'{k}={v}' for k, v in self.__dict__.items() if k not in {'episode', 'step'})
        return f'episode={self.episode}, step={self.step}, \nstate=({state})'


class ResponseStateManager:
    def __init__(self):
        self.episode = 0
        self.step = 0
        self.response_step_obs = None
        self.response_state = None
        self.show_packets = False
        self.reset_episode()

    def reset_episode(self)->(ResponseStepObservations, ResponseState):
        self.episode += 1
        self.step = 0
        self.response_state = ResponseState(self.episode, self.step)
        self.response_step_obs = ResponseStepObservations(self.episode, self.step)
        return self.response_step_obs, self.response_state

    def begin_next_step(self)->(ResponseStepObservations, ResponseState):
        previous_state = self.response_step_obs
        self.step += 1
        self.response_step_obs = ResponseStepObservations(self.episode, self.step)
        return previous_state, self.response_state

    def set_llm_preview(self, llm_preview):
        self.response_step_obs.llm_preview = llm_preview
        self.response_state.llm_preview = llm_preview

    def add_llm_response_and_clear_llm_preview(self, llm_response):
        self.response_state.current_responses.append(llm_response)
        self.response_state.speech_chunks_per_response.append(0)
        self.response_step_obs.llm_responses.append(llm_response)
        self.response_step_obs.llm_preview = ''
        self.response_state.llm_preview = ''

    def add_tts_raw_chunk_id(self, chunk_id, llm_sentence_id):
        self.response_state.speech_chunks_per_response[llm_sentence_id] += 1
        self.response_step_obs.tts_raw_chunk_ids.append(chunk_id)

    def pretty_print_current_responses(self)->str:
        line = ""
        for i, response in enumerate(self.response_state.current_responses):
            line += "🤖 " if len(line) == 0 else ""
            line += f"[{self.response_state.speech_chunks_per_response[i]}] " if self.show_packets else ""
            line += f"{response}  \n"
        return line
    
    def pretty_print_preview_text(self)->str:
        robot_preview_text = ""
        if len(self.response_state.llm_preview):
                robot_preview_text = f"🤖❓ {self.response_state.llm_preview}"
        return robot_preview_text
