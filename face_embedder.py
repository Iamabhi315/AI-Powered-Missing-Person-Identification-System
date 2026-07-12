import os
import pickle
import insightface
import config

class FaceEmbedder:
    def __init__(self):
        print("[INFO] Loading InsightFace Embedder...")
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        try:
            self.app.prepare(ctx_id=0) # GPU
        except Exception:
            self.app.prepare(ctx_id=-1) # CPU

    def get_embedding(self, face_crop):
        """
        Takes cropped face and returns 512-D embedding

        """
        faces = self.app.get(face_crop)
        if len(faces) == 0:
            return None
        return faces[0].embedding

    def load_saved_embeddings(self):
        """
        Loads embedding dictionary from saved pikle

        """
        if os.path.exists(config.EMBEDDING_FILE):
            with open(config.EMBEDDING_FILE, "rb") as f:
                return pickle.load(f)
        return {}

    def save_embeddings(self, embeddings_dict):
        """
        Save Dictionary to pickle file 
        
        """
        os.makedirs(config.EMBEDDING_DIR, exist_ok=True)
        with open(config.EMBEDDING_FILE, "wb") as f:
            pickle.dump(embeddings_dict, f)