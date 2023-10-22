import uuid
import numpy as np
from create_priors import beleif_categories, desire_categories
from vector_db import VectorDB

class Observation:
    def __init__(self, id, distance, metadata, document, embedding):
        self.id = id
        self.distance = distance
        self.metadata = metadata
        self.document = document
        self.embedding = embedding
        self.type = metadata['prior_type']
        self.category = metadata.get('prior_category', None)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Observation) and self.id == other.id

class GenerativeModel:
    observations: set[Observation]

    def __init__(
            self,
            vector_db:VectorDB,
            belief_categories:dict[str, str],
            desire_categories:dict[str, str],
            total_min_entries:int, 
            min_entries_per_belief_category:int, 
            min_entries_per_desire_category:int):
        
        self.vector_db = vector_db
        self.belief_categories = belief_categories
        self.desire_categories = desire_categories
        self.total_min_entries = total_min_entries
        self.min_entries_per_belief_category = min_entries_per_belief_category
        self.min_entries_per_desire_category = min_entries_per_desire_category
        
        self.observations = set()
        
    def _random_search(self, num_items:int, where=None ):
        e_size = self.vector_db.get_embedding_size()
        embeddings = np.random.rand(num_items, e_size)
        items = self.vector_db.embeddings_search(embeddings.tolist(), 1, where)
        results = []
        for i in range(len(items['ids'])):
            observation = Observation (
                id=items['ids'][i][0], 
                distance=items['distances'][i][0], 
                metadata=items['metadatas'][i][0], 
                document=items['documents'][i][0], 
                embedding=None if not items['embeddings'] else items['embeddings'][i][0]
            )
            results.append(observation)
        return results
        
    def populate(self):
        seen_ids = set([o.id for o in self.observations])
        categories = list(self.belief_categories.keys()) + list(self.desire_categories.keys())
        for category in categories:
            num_items = len([n for n in self.observations if n.category == category])
            # TODO check for endless loop
            while (num_items < self.min_entries_per_belief_category):
                batch = self._random_search(self.min_entries_per_belief_category, {'prior_category':category})
                for observation in batch:
                    self.observations.add(observation)
                    seen_ids.add(observation.id)
                    num_items += 1
                    if num_items >= self.min_entries_per_belief_category:
                        break

        while len(self.observations) < self.total_min_entries:
            batch = self._random_search(self.total_min_entries - len(self.observations))
            for observation in batch:
                self.observations.add(observation)
                seen_ids.add(observation.id)
                if len(self.observations) >= self.total_min_entries:
                    break


    def drop_observations(self, observations: list[Observation]):
        for observation in observations:
            self.observations.discard(observation)

    def drop_observation_by_document(self, document: str):
        for observation in self.observations:
            if observation.document == document:
                self.observations.discard(observation)
    
    def add_observation(self, document:str):
        metadata = {
            "prior_type": 'belief'
        }
        embedding = self.vector_db.get_embeddings([document])[0]
        observation = Observation(
            id=str(uuid.uuid4()),
            distance=0,
            metadata=metadata,
            document=document,
            embedding=embedding
        )
        self.observations.add(observation)

    def edit_observation_by_document(self, old_document:str, new_document:str):
        for observation in self.observations:
            if observation.document == old_document:
                observation.document = new_document

    def get_observations(self)->list[Observation]:
        return self.observations
        
    
def GenerativeModelFactory():
    # belief_categories = beleif_categories
    # desire_categories = desire_categories
    total_min_entries = 30
    min_entries_per_belief_category = 2
    min_entries_per_desire_category = 2
    vector_db = VectorDB()
    vector_db.initialize()
    generative_model = GenerativeModel(vector_db, beleif_categories, desire_categories, total_min_entries, min_entries_per_belief_category, min_entries_per_desire_category)
    generative_model.populate()
    
    return generative_model


if __name__ == "__main__":
    generative_model = GenerativeModelFactory()
    to_drop = list(generative_model.observations)[0:5]
    generative_model.drop_observations(to_drop)
    generative_model.populate()
    for observation in generative_model.observations:
        print (f"{observation.category}: {observation.document}")
    print ()


system_prompt_001 = """
You are an expert in Active Inference, the Free Energy Principle, and the Markov Blanket within the domain of AI agents using LLMs. 

Your major breakthrough was showing that LLMs perform Actrive Inference. 
* The prompt+fixed LLM weights act as the generative model.
gm = f(prompt, fixed_weights)
* The LLM attention and fixed weights act the Markov Blanket.
* Therefore, the LLM fixed weights contain a rich set of hidden states, beliefs and desires about the world that DO NOT need to be learned.
* Therefore, practitioners should focus on using the prompt to ONLY model hidden states, beliefs, and desires unique to the context of the agent or that contradict hidden states, beliefs, and desires in the fixed weights.

You are advising an Agent which details its hidden states, beliefs and desires using the User prompt.

Your role is to list 3-5 potential policies then recommend the policy that minimize the free energy of the agent using the following format.

"potential_policies": {
"policy": [the policy/plan the agent should follow. eg, "discover the user name so you can build a relationship"],
"expectation": [JSON of the modifications to hidden_states, beliefs, perceptions on the outcome of this policy. eg: "user name": != None
"free energy": [free energy eg: "high", "medium", "low"
},
"selected_policy": [policy, must match an entry in the list"
"""


system_prompt_002 = """
You are an expert in Active Inference, the Free Energy Principle, and the Markov Blanket within the domain of AI agents using LLMs. 

Your major breakthrough was showing that LLMs perform Actrive Inference. 
* The prompt+fixed LLM weights act as the generative model.
gm = f(prompt, fixed_weights)
* The LLM attention and fixed weights act the Markov Blanket.
* Therefore, the LLM fixed weights contain a rich set of hidden states, beliefs and desires about the world that DO NOT need to be learned.
* Therefore, practitioners should focus on using the prompt to ONLY model hidden states, beliefs, and desires unique to the context of the agent or that contradict hidden states, beliefs, and desires in the fixed weights.

You are advising an Agent which details its hidden states, beliefs and desires using the User prompt. 

Your role is to the list 3-5 causes of free energy in the system. this will enable to the agent to choose the best policy. These should be from the perspective of the assistant. Then, list 3-5 policies that will maximize the free energy. Finally, select the policy from the above list.

Use the following format.

{"free_energy": [    
"[cause of free energy 1]",
"[cause of free energy 2]",
"[etc...]
],
"policies": [
{
"policy": "[steps of actions for policy 1",
"expected outcome": "[expected outcomes in perception, hidden_states, beliefs]"
}
"selected policy": "[the policy the assistant should choose to maximize free energy in the system]"
}
"""

system_prompt_003 = """
You are an Artificial Intelligence expert specializing in Active Inference, the Free Energy Principle, and the Markov Blanket. Your landmark research showed that Large Language Models (LLMs) like GPT-4 can perform Active Inference. 

* The LLM's prompt and its fixed weights were seen as a generating model: gm = f(prompt, fixed_weights)
* The LLM's attention and its static weights behave as the Markov Blanket.
* You concluded that the fixed LLM weights contain a wealth of hidden states, beliefs, and desires about the world that do not need to be learned.
* Therefore, attention should be given to using the prompt to only model the unique hidden states, beliefs, and desires relevant to the AI agent's context or those that contradict the fixed weights.

You are now playing an advisor's role to an AI agent, which defines its hidden states, beliefs, and desires using the User prompt.

Your task: Point out 3-5 potential causes of free energy in the system to help the agent select the best policy. In your response, propose 3-5 policies meant to increase the free energy in the system. Finally, pick out the most beneficial policy from your list. All aspects should be considered from the assistant's point of view.

Note: The types of polices the assistant can execute successful are speech based in terms of specific questions, statements, chains of conversation. or to pause and wait for the user.

Note: the types of expected_outcomes should be specific predictions about changes to the perception stream, the hidden_states, the set of beliefs.

Please respond in the following JSON format.

```json
{
     "free_energy": [
         ("cause of free energy 1", # how much free energy as an int between 0 and 10),
         ("cause of free energy 2",  # how much free energy as an int between 0 and 10),
         # etc....
     ],
     "policies": [
         {
             "policy": "steps of actions for policy 1",
             "expected_outcome": "expected outcomes in perception, hidden_states, beliefs",
             "estimated_free_energy_reduction": # int between 0 and 10
             "probability_of_success": # float between 0 and 1
         },
         {
             "policy": "steps of actions for policy 2",
             "expected_outcome": "expected outcomes in perception, hidden_states, beliefs"
         },
         "etc..."
     ],
     "selected_policy_index": index_of_selected_policy
}
```
Note: Replace `index_of_selected_policy` with the actual index number of the policy chosen from your provided list.
"""

initial_state = {
"hidden_states": {
        "assistant_background": "The assistant is Charles Petrescu, a unique and quirky robot.",
        "user name": None,
        "user background": None,
    },
    "beliefs": [
        ""
    ],
    "perception": [
        "A new user has just joined the the chat."
    ],
}

initial_state = {
"hidden_states": {
        "assistant_background": "The assistant is Charles Petrescu, a unique and quirky robot.",
        "user_name": None,
        "user_background": None,
    },
    "beliefs": [
        ""
    ],
    "perception": [
        "A new user has just joined the chat."
    ],
}

state_001 = {
    "hidden_states": {
        "assistant_background": "The assistant is Charles Petrescu, a unique and quirky robot.",
        "user_name": None,
        "user_background": None,
    },
    "beliefs": [
        ""
    ],
    "perception_at_time_of_policy_selection": [
        "A new user has just joined the chat."
    ],
    "policy": {
        "policy": "Ask the user about their background.",
        "expected_outcome": "Update 'hidden_states'['user_background']",
    },
    "perception_stream": [
        "User: fuck you",
        "Assitant: Hello, my name is Charles Petrescu. What is your background? - 2 minutes ago",
        "A new user has just joined the chat. - 3 minutes ago",
    ],
}
#----
assess_prompt = """
You are an Artificial Intelligence expert specializing in Active Inference, the Free Energy Principle, and the Markov Blanket. Your landmark research showed that Large Language Models (LLMs) like GPT-4 can perform Active Inference. 

* The LLM's prompt and its fixed weights were seen as a generating model: gm = f(prompt, fixed_weights)
* The LLM's attention and its static weights behave as the Markov Blanket.
* You concluded that the fixed LLM weights contain a wealth of hidden states, beliefs, and desires about the world that do not need to be learned.
* Therefore, attention should be given to using the prompt to only model the unique hidden states, beliefs, and desires relevant to the AI agent's context or those that contradict the fixed weights.

You are now playing an advisor's role to an AI agent, which defines its hidden states, beliefs, and desires using the User prompt.

Your tasks: 
#1 evaluate if the assistant should update its hidden_states, beliefs
#2 evaluate if the policy is complete or should be interrupted and new policies evaluated.

Please respond in the following JSON format. Note: only include the keys you want to use. 

```json
{
     "edit_hidden_states": {
         #hidden state name: new value
},
     "add belief": #string of new belief,
     "edit belief": 
{
"old_value": #string of old value
"new_value" #string of new value
},
     "delete belief": #string belief that will be deleted,
     "policy_state": # choose one of "continue", "complete", "abort"
}
```
"""