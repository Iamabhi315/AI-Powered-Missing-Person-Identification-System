import sqlite3
import pickle
import os
import shutil
import config

DB_FOLDER = "known_faces"
DB_NAME = "missing_persons.db"


def _get_embedding_file_path():
    return config.EMBEDDING_FILE


def _load_embeddings_from_file():
    embedding_file = _get_embedding_file_path()
    if os.path.exists(embedding_file):
        with open(embedding_file, "rb") as f:
            return pickle.load(f)
    return {}


def _save_embeddings_to_file(embeddings_dict):
    embedding_file = _get_embedding_file_path()
    os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
    with open(embedding_file, "wb") as f:
        pickle.dump(embeddings_dict, f)


def _migrate_embeddings_from_db():
    embeddings_dict = _load_embeddings_from_file()
    if embeddings_dict:
        return embeddings_dict

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT full_name, embedding FROM missing_persons")
    rows = cursor.fetchall()
    conn.close()

    embeddings_dict = {}
    for row in rows:
        name = row[0]
        embedding_bytes = row[1]
        if embedding_bytes:
            embedding_array = pickle.loads(embedding_bytes)
            if name not in embeddings_dict:
                embeddings_dict[name] = []
            embeddings_dict[name].append(embedding_array)

    _save_embeddings_to_file(embeddings_dict)
    return embeddings_dict

def init_db():
    """
    Creates Database (DDL Operation)

    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Creating table of missing persons
    # Save Embedding in BLOB(binary large object)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS missing_persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            gender TEXT,
            email TEXT,
            age INTEGER,
            address TEXT,
            embedding BLOB 
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[INFO] Database initialized successfully.")

def insert_person(full_name, gender, email, age, address, embedding_array):
    """
    Insert data of new missing person
    
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # To save numpy array(embedding) in database , convert into bytes/pickle
    embedding_bytes = pickle.dumps(embedding_array)
    
    cursor.execute('''
        INSERT INTO missing_persons (full_name, gender, email, age, address, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (full_name, gender, email, age, address, embedding_bytes))
    
    conn.commit()
    conn.close()

    embeddings_dict = _load_embeddings_from_file()
    if full_name not in embeddings_dict:
        embeddings_dict[full_name] = []
    embeddings_dict[full_name].append(embedding_array)
    _save_embeddings_to_file(embeddings_dict)

def get_all_embeddings():
    """
    Load embeddings for face recognition from the pickle file.

    """
    return _migrate_embeddings_from_db()

# Table will be created when it run
if __name__ == "__main__":
    init_db()

# to get email

def get_email_by_name(name):
    """
    Reads email from the database on the basis of Name

    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT email FROM missing_persons WHERE full_name=?", (name,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result and result[0]:
        return result[0]
    return None


def get_all_person_names():
    """Return all stored person names from the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT full_name FROM missing_persons ORDER BY full_name")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows if row[0]]


def delete_person(full_name):
    """Delete a person record from the database by full name and remove the saved face images."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM missing_persons WHERE full_name=?", (full_name,))
    conn.commit()
    deleted_rows = cursor.rowcount
    conn.close()

    if deleted_rows > 0:
        embeddings_dict = _load_embeddings_from_file()
        if full_name in embeddings_dict:
            del embeddings_dict[full_name]
            _save_embeddings_to_file(embeddings_dict)

        person_folder = os.path.join(DB_FOLDER, full_name)
        if os.path.isdir(person_folder):
            shutil.rmtree(person_folder, ignore_errors=True)
        return True
    return False