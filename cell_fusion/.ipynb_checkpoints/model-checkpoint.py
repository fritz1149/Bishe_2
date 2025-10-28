import torch.nn as nn
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
# import sys
model_dir = '/home/changc/Bishe/cell_fusion'
# sys.path.insert(0, model_dir)
# print(sys.path)

from encoder import CustomEncoder
import fire

max_length = 8192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.body_llm = AutoModelForCausalLM.from_pretrained(f'{model_dir}/{model_path}')
        self.cell_encoder = CustomEncoder(model_path)
        self.embed_tokens = self.body_llm.model.embed_tokens

    def forward(self, text_input_ids, attention_mask, cell_input_ids, labels, cell_pos):
        text_input_ids = text_input_ids.to(device)
        attention_mask = attention_mask.to(device)
        cell_input_ids = cell_input_ids.to(device)
        labels = labels.to(device)
        cell_pos = cell_pos.to(device)
        
        text_embeddings = self.embed_tokens(text_input_ids)
        cell_embeddings = self.embed_tokens(cell_input_ids)
        cell_embeddings = self.cell_encoder(cell_embeddings)

        embeddings = text_embeddings
        for i in range(embeddings.shape[0]):
            embeddings[i, cell_pos[i].item()] = cell_embeddings[i]
        output = self.body_llm.forward(attention_mask=attention_mask,
                                       inputs_embeds=embeddings,
                                       labels=labels)
        return output

def train(model_path, if_train, log_step):
    print('if_train', if_train)
    if if_train == 1:
        model = CustomModel(model_path).to(device)
        frozen_modules = ['body_llm', 'embed_tokens']
        for name, param in model.named_parameters():
            for frozen_module in frozen_modules:
                if frozen_module in name:
                    param.requires_grad = False
                    break
    
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="test.json")

    tokenizer = AutoTokenizer.from_pretrained(f'{model_dir}/{model_path}', padding_side='left')
    tokenizer.add_tokens("[cell]", True)
    cell_token = tokenizer.encode("[cell]")[0]
    import numpy as np
    def map_func(example):
        # 应用 chat_template
        texts = []
        for prompt, output in zip(example["prompt"], example["output"]):
            conversation = [{"role": "user", "content": f"{prompt}: [cell] \\nothink"}, {"role": "assistant", "content": output}]
            text = tokenizer.apply_chat_template(conversation, tokenize=False)
            texts.append(text)
        text_batch_dict = tokenizer(texts, return_tensors="pt", padding=True)
        cell_batch_dict = tokenizer(example["cell"], return_tensors="pt")
        text_input_ids = text_batch_dict["input_ids"]

        response_template = "<|im_start|>assistant\n"
        response_prefix_input_ids = tokenizer.encode(response_template)
        response_prefix_input_ids = torch.Tensor(response_prefix_input_ids)
        ignore_index = -100
        labels = text_input_ids.clone()
        for i in range(labels.shape[0]):
            for j in np.where(labels[i] == response_prefix_input_ids[0])[0]:
                # print(i, j, j+response_prefix_input_ids.shape[0], np.where(labels[i] == response_prefix_input_ids[0]))
                if torch.equal(labels[i, j:j+response_prefix_input_ids.shape[0]], response_prefix_input_ids):
                    labels[i,:j+response_prefix_input_ids.shape[0]] = ignore_index
                    break #仅考虑单论对话的情况
        cell_pos = [np.where(text_input_ids[i] == cell_token)[0] for i in range(text_input_ids.shape[0])]
        cell_pos = torch.Tensor(cell_pos).to(torch.int)
        ret = {
                "text_input_ids": text_input_ids, 
                "attention_mask": torch.cat((text_batch_dict["attention_mask"], torch.ones(text_batch_dict["attention_mask"].shape[0], 1)), dim=1),
                "cell_input_ids": cell_batch_dict["input_ids"],
                "labels": labels,
                "cell_pos": cell_pos,
               }
        # for k,v in ret.items():
        #     print(k, type(v), v.shape)
        return ret

    dataset = dataset.map(
        map_func,
        batched=True,
        remove_columns=['prompt', 'output', 'cell'],
        batch_size=1000,
        num_proc=1,
        keep_in_memory=True,  # 测试数据集不需要写入缓存
    )
    dataset.set_format(type="torch", columns=["text_input_ids", "attention_mask", "cell_input_ids", "labels", "cell_pos"])
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset['train'],
        batch_size=1,
    )

    if if_train == 0:
        for i, x in enumerate(dataloader):
            print(i)
            for k,v in x.items():
                print(k, v)
            
    if if_train == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        for epoch in range(10):
            for step, x in enumerate(dataloader):
                output = model(**x)
                loss = output.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % log_step == 0:
                    print(step, loss)
                    
def main(if_train: int = 0, log_step: int = 20):
    train('Qwen3-4B', if_train)
    
if __name__ == '__main__':
    fire.Fire(main)