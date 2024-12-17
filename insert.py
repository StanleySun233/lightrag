import os
from lightrag import LightRAG
from lightrag.llm import gpt_4o_mini_complete
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./data"


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
#     llm_model_name='Qwen/Qwen2.5-1.5B',  # Model name from Hugging Face
#     # Use Hugging Face embedding function
#     embedding_func=EmbeddingFunc(
#         embedding_dim=1536,
#         max_token_size=5000,
#         func=lambda texts: hf_embedding(
#             texts,
#             tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
#             embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#         )
#     ),
# )

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

for i in os.listdir("./book"):
    print(i)
    with open(f"./book/{i}",encoding='utf-8') as f:
        rag.insert(f.read())
    # spend: 0.23USD for 7documents.