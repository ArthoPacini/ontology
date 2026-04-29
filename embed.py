import struct
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Load a lightweight, highly capable local model for semantic search
# "all-MiniLM-L6-v2" is fast and produces 384-dimensional embeddings
try:
    logger.info("Loading local embedding model (SentenceTransformer)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Local embedding model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    model = None

def get_embedding(text: str) -> bytes:
    """
    Returns the embedding as a flat binary array of 32-bit floats.
    sqlite-vec expects little-endian float32 bytes.
    """
    if model is None:
        return b""
        
    if not text or not text.strip():
        text = "empty"
        
    # Create numpy embedding
    vector = model.encode(text)
    
    # Pack into little-endian 32-bit floats for sqlite-vec
    return struct.pack(f"<{len(vector)}f", *vector)

def get_embedding_list(text: str) -> list[float]:
    """Returns standard python list of floats"""
    if model is None:
        return []
    if not text or not text.strip():
        text = "empty"
    vector = model.encode(text)
    return vector.tolist()
