import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

embedding_model = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
device = "cpu"

def get_vector_store(index_name: str) -> RedisVectorStore:
    """Initializes and returns a RedisVectorStore instance."""
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    redis_config = RedisConfig(
        index_name=index_name,
        redis_url=redis_url,
    )
    
    return RedisVectorStore(
        embeddings=embeddings, 
        config=redis_config
    )
