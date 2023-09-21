import json

class AgentResponse(dict):
    def __init__(self, prompt=None, **kwargs):
        super().__init__(**kwargs)
        if prompt is not None:
            self['prompt'] = prompt
            self['llm_preview'] = ''
            self['llm_sentence'] = ''
            self['llm_sentence_id'] = 0
            self['llm_sentences'] = []
            self['tts_raw_chunk_ref'] = None
            self['tts_raw_chunk_id'] = 0

    def make_copy(self):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.update(self.copy())
        return new_instance

    def to_json(self):
        return json.dumps(self)
