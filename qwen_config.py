from transformers import AutoConfig

# 加载模型配置
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B")

# 打印 embedding 维度
print("Embedding Dimension:", config.hidden_size)
