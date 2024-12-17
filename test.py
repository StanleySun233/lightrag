import os
from lightrag import LightRAG
from lightrag.llm import gpt_4o_mini_complete
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./data"


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
    llm_model_name='Qwen/Qwen2.5-3B',  # Model name from Hugging Face
    # Use Hugging Face embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=5000,
        func=lambda texts: hf_embedding(
            texts,
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        )
    ),
)

# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
#     # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
# )

print(rag.query("How many metres of under-keel clearance should be maintained throughout the passage through the Straits of Malacca and Singapore?"),"global")
# time spend: 6mins for Qwen/Qwen2.5-3B-Instruct at 2060-6g