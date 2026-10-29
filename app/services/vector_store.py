import numpy as np
import faiss 

def store_vec(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32")) #faiss expects float32
    return index

def index_search(index,question_embedding,k=5):
    distances, indices = index.search(
        question_embedding.astype("float32"),
        k
    )
    return indices