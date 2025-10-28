from datasets import load_dataset
ds = load_dataset("json", data_files="test.json")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen3-4B', padding_side='left')

def map_func(example):
    # 应用 chat_template
    texts = []
    for prompt, output in zip(example["prompt"], example["output"]):
        conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]
        text = tokenizer.apply_chat_template(conversation, tokenize=False)
        # print(text)
        texts.append(text)
    print(type(example["cell"]))
    # 应用 tokenizer
    output = tokenizer(texts, return_tensors="pt")
    return {'input_ids': output['input_ids'], 'attention_mask': output['attention_mask']}

ds = ds.map(
    map_func,
    batched=True,
    # remove_columns='conversation',
    batch_size=1000,
    num_proc=1,
    keep_in_memory=True,  # 测试数据集不需要写入缓存
)

# from trl import DataCollatorForCompletionOnlyLM
# response_template = "<|im_start|>assistant\n"
# collator = DataCollatorForCompletionOnlyLM(
#     tokenizer=tokenizer,
#     response_template=response_template,
# )
# def collate_fn(batch):
#     batch = collator(batch)

from torch.utils.data import DataLoader
dataloader = DataLoader(
    ds['train'],
    batch_size=1,
    # collate_fn=collate_fn,
)
for i,  x in enumerate(dataloader):
    print(i, end=": ")
    for e in x:
        print(type(e), end=" ")
    print("")