import asyncio
import json
import os
import sys
import traceback
import chromadb
from chromadb.utils import embedding_functions
import uuid
import numpy as np

class VectorDB:
    def __init__(self):
        self._initialized = False
        self.client = chromadb.PersistentClient(path=os.path.join(".", "vector_db"))
        self.beliefs_and_desires_collection = self.client.get_or_create_collection(
            name="beliefs_and_desires",
            metadata={"hnsw:space": "l2"}
            # metadata={"hnsw:space": "cosine"} # l2 is the default
            )
        self.embed = embedding_functions.DefaultEmbeddingFunction()

    def initialize(self):
        if self.beliefs_and_desires_collection.count() == 0:
            self.init_beilefs_and_desires()
        self._embedding_size = len(self.embed(["test"])[0])
        self._initialized = True
   
    def init_beilefs_and_desires(self):
        # load beliefts from priors/beliefs.json
        beliefs_file_path = os.path.join('priors', 'beliefs.json')
        with open(beliefs_file_path) as json_file:
            beliefs = json.load(json_file)
        desires_file_path = os.path.join('priors', 'desires.json')
        with open(desires_file_path) as json_file:
            desires = json.load(json_file)
        b_and_d = beliefs | desires
        total = sum(len(b_and_d[key]) for key in b_and_d)
        print(f"Adding {total} beliefs and desires to beliefs_and_desires_collection...")
        i = 0
        for category in b_and_d.keys():
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            category_lower = category.lower()
            prior_type = "belief" if "belief" in category_lower else "desire" if "desire" in category_lower else None
            metadata = {"prior_category": category, "prior_type": prior_type} if prior_type else {"prior_category": category}
            for value in b_and_d[category]:
                percentage = (i + 1) / total * 100
                progress = int(percentage // (100/60))
                bar = f"[{'#' * progress}{'.' * (60 - progress)}] {percentage:.2f}%"
                print(f"\r{bar}", end="")
                documents.append(value)
                metadatas.append(metadata)
                e = self.embed([value])[0]
                embeddings.append(e)
                ids.append(str(uuid.uuid4()))
                i += 1
            try:
                self.beliefs_and_desires_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            except Exception as e:
                print(f"Error adding {category} to beliefs_and_desires_collection: {e}")
                trace = traceback.format_exc()
                print(f"trace: {trace}")
                raise e
        print()

    def get_embedding_size(self):
        assert self._initialized, "VectorDB not initialized. Call initialize() first."
        return self._embedding_size

    def search(self, query_text, n_results=5, prior_category=None, prior_type=None):
        assert self._initialized, "VectorDB not initialized. Call initialize() first."
        assert prior_category is None and prior_type is None, "Not implemented yet."
        sorted_results = self.beliefs_and_desires_collection.query(
            query_texts=query_text,
            n_results=n_results,
        )
        return sorted_results
    
    def embeddings_search(self, embeddings, n_results=5, where:chromadb.Where = None):
        assert self._initialized, "VectorDB not initialized. Call initialize() first."
        sorted_results = self.beliefs_and_desires_collection.query(
            query_embeddings=embeddings,
            n_results=n_results,
            where=where
        )
        return sorted_results
    
    def get_embeddings(self, document:str):
        assert self._initialized, "VectorDB not initialized. Call initialize() first."
        embeddings = self.embed(document)
        return embeddings

if __name__ == "__main__":
    vector_db = VectorDB()
    vector_db.initialize()

    print()

    def search(str):
        sorted_results = vector_db.search(str)
        print(f"'{str}' results:")
        for i in range(len(sorted_results['ids'][0])):
            # print(f" {sorted_results['distances'][0][i]:.5f}, '{sorted_results['documents'][0][i]}', '{sorted_results['ids'][0][i]}', {sorted_results['metadatas'][0][i]}")
            print(f" - {sorted_results['distances'][0][i]:.5f}, '{sorted_results['documents'][0][i]}', {sorted_results['metadatas'][0][i]['prior_category']}")
        print()

    if len(sys.argv) < 2:
        # no arguments, so use some defaults
        search("I like cabbages")
        search("I live traveling")
        search("I am sad")
        exit()

    str = " ".join(sys.argv[1:])
    search(f"the user said: {str}")


