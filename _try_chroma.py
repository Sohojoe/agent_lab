import os
import chromadb
from chromadb.utils import embedding_functions
import uuid
import numpy as np


# client = chromadb.Client()
client = chromadb.PersistentClient(path=os.path.join(".", "chromadb"))


collection = client.get_or_create_collection(
    name="my_collection",
    metadata={"hnsw:space": "l2"}
    # metadata={"hnsw:space": "cosine"} # l2 is the default
    )

sample_text = [
    "I have a dog named Loba",
    "I have a dog named Alife",
    "Loba is blind",
    "Loba is missing an eye",
    "Alfie is a happy dog",
    "Sam is happy",
    "Sam is sad",
    "Sam is hungry",
    "Sam is tired",
    "Sam is angry",
    "Sam is sleepy",
    "Sam is awake",
]


if collection.count() < len(sample_text):
    ids = [str(uuid.uuid4()) for _ in sample_text]
    metadata = [{"source": "my_source"} for _ in sample_text]
    collection.add(
        documents=sample_text,
        metadatas=metadata,
        ids=ids
    )

def search(query_text, n_results=5):
    sorted_results = collection.query(
        query_texts=query_text,
        n_results=n_results
    )

    print(f"'{query_text}' results:")
    for i in range(len(sorted_results['ids'][0])):
        print(f" {sorted_results['distances'][0][i]:.5f}, '{sorted_results['documents'][0][i]}', '{sorted_results['ids'][0][i]}', {sorted_results['metadatas'][0][i]}")
    print()

def search_embeddings(embeddings, n_results=2):
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()
    sorted_results = collection.query(
        query_embeddings=embeddings,
        n_results=n_results,
        include=["embeddings", "metadatas", "documents", "distances"]
    )

    print(f"results:")
    for i in range(len(sorted_results['ids'][0])):
        print(f" {sorted_results['distances'][0][i]:.5f}, '{sorted_results['documents'][0][i]}', '{sorted_results['ids'][0][i]}', {sorted_results['metadatas'][0][i]}")
    print()


search("Loba")
search("Alfie")
search("dog")
search("blind")
search("my mood")

default_ef = embedding_functions.DefaultEmbeddingFunction()
happy = np.array(default_ef(["happy"]))
dog = np.array(default_ef(["dog"]))
alfie = np.array(default_ef(["Alfie"]))
happy_dog = happy + dog
happy_dog = happy_dog / np.linalg.norm(happy_dog)
happy_dog_alfie = happy + dog + alfie
happy_dog_alfie = happy_dog_alfie / np.linalg.norm(happy_dog_alfie)

search_embeddings(happy)
search_embeddings(dog)
search_embeddings(alfie)
search_embeddings(happy_dog)
search_embeddings(happy_dog_alfie)

print ("--done--")