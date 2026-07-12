import numpy as np
from numpy.linalg import norm
import config

class FaceRecognizer:
    def __init__(self, saved_embeddings):
        # Stores embeddings like database
        self.saved_embeddings = saved_embeddings

    def cosine_similarity(self, emb1, emb2):
        """
        Calculates similarity score between two embeddings
        
        """
        return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))

    def identify_person(self, new_embedding):
        """
         Compares New embedding and return best match
        
        """
        best_match_name = "Unknown"
        highest_similarity = -1

        for person_name, embeddings_list in self.saved_embeddings.items():
            for saved_emb in embeddings_list:
                sim_score = self.cosine_similarity(new_embedding, saved_emb)
                
                if sim_score > highest_similarity:
                    highest_similarity = sim_score
                    if highest_similarity > config.SIMILARITY_THRESHOLD:
                        best_match_name = person_name

        return best_match_name, highest_similarity