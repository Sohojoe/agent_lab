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
        self.category = metadata['prior_category']

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
