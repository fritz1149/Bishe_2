from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen3-4B', padding_side='left')
print(tokenizer.decode([151667]))