from typing import Iterator
from torch.nn import Parameter
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
import sys

def get_llm(args):
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.llm, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(f"./{args.llm}")
    return model

class EditedQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self):
        super().__init__()
        self.model.visual = torch.nn.Identity()
    def forward(self, **kwargs):
        if kwargs.get("inputs_embeds") is not None and kwargs.get("input_ids") is not None:
            del kwargs["input_ids"]
        return super().forward(**kwargs)

class ProposeModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        backbone = get_llm(args)
        backbone.model.visual = torch.nn.Identity()
        self.text_embedder = backbone.model.get_input_embeddings()
        
        from peft import LoraConfig, PeftMixedModel, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        backbone = PeftMixedModel(backbone, lora_config, adapter_name="0")
        from encoder import get_longformer_with_projector
        self.encoder = get_longformer_with_projector(args)
        if args.align1_mode:
            self.backbone = backbone
            # # 冻结除lora参数以外的所有参数
            for name, param in self.named_parameters(recurse=True):
                if "lora_A.0" not in name and "lora_B.0" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            self.backbone.gradient_checkpointing_enable()
            self.backbone.enable_input_require_grads()
        elif args.align2_mode:
            # 对齐模式下，冻结除了 self.encoder.fc 之外的所有层
            self.backbone = backbone
            for name, param in self.named_parameters(recurse=True):
                if "encoder.fc" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # 只开放 self.encoder.fc 可学习
            for param in self.encoder.fc.parameters():
                param.requires_grad = True
            self.backbone.gradient_checkpointing_enable()
            self.backbone.enable_input_require_grads()
        elif args.finetune_mode:
            # 微调等其他模式，使用peft定义lora，并冻结除lora模块的其他参数
            backbone.add_adapter("1", lora_config)
            backbone.set_adapter(["0", "1"])
            self.backbone = backbone
            # # 冻结除lora参数以外的所有参数
            for name, param in self.named_parameters(recurse=True):
                if "lora_A.1" not in name and "lora_B.1" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            self.backbone.gradient_checkpointing_enable()
            self.backbone.enable_input_require_grads()
        elif args.eval_mode:
            backbone.add_adapter("1", lora_config)
            backbone.set_adapter(["0", "1"])
            self.backbone = backbone
            for name, param in self.named_parameters(recurse=True):
                param.requires_grad = False
            # self.backbone.gradient_checkpointing_enable()
            # self.backbone.enable_input_require_grads()
        # for name, param in self.named_parameters(recurse=True):
        #     if param.requires_grad:
        #         print(name)
        # for name, module in self.named_modules():
        #     print(name, type(module).__name__)
        # for name, param in self.named_parameters(recurse=True):
        #     print(name)

    def parameters_(self, recurse: bool = True) -> Iterator[Parameter]:
        return [p for p in self.parameters(recurse=recurse) if p.requires_grad]
    def state_dict_(self, args):
        d = self.state_dict()
        # 使用字典推导式创建新字典，只保留需要梯度的参数
        # 注意：state_dict() 返回的是 Tensor，需要检查原始参数的 requires_grad
        # 但由于 state_dict 中的值已经是 detached 的，我们需要通过参数名来判断
        result = {}
        for name, param in self.named_parameters(recurse=True):
            if param.requires_grad:
                result[name] = d[name]
        return result

    def forward(self, x):
        input_ids, labels_ids, payloads, position_ids, attention_mask = x
        IMAGE_PAD_ID = 151655
        text_device = next(self.text_embedder.parameters()).device
        encoder_device = next(self.encoder.parameters()).device

        input_ids = input_ids.to(text_device)
        input_embeddings = self.text_embedder(input_ids)
        # Clone input_embeddings to avoid in-place operation on leaf variable that requires grad
        input_embeddings = input_embeddings.clone()

        for i, payload_tuples in enumerate(payloads):
            if payload_tuples is None:
                input_payload_pos = input_ids[i] == IMAGE_PAD_ID
                assert input_payload_pos.sum().item() == 0
                continue
            payload_ids, attention_mask_payload, global_attention_mask_payload = payload_tuples
            payload_ids = payload_ids.to(encoder_device)
            attention_mask_payload = attention_mask_payload.to(encoder_device)
            global_attention_mask_payload = global_attention_mask_payload.to(encoder_device)
            # print(payload_ids.shape)
            # sys.stdout.flush()
            payload_embeddings = self.encoder((payload_ids, attention_mask_payload, global_attention_mask_payload))
            payload_embeddings = payload_embeddings.squeeze(1).to(text_device)
            # print(payload_embeddings.shape)
            input_payload_pos = input_ids[i] == IMAGE_PAD_ID
            # print(input_ids[i].shape, input_payload_pos.shape, input_payload_pos.sum().item())
            assert payload_embeddings.shape[0] == input_payload_pos.sum().item()
            input_embeddings[i, input_payload_pos] = payload_embeddings
        # mrope_position_deltas = position_ids.max(dim=1).values+1-input_ids.shape[1] # [batch_size]
        # mrope_position_deltas = mrope_position_deltas.to(backbone_device)

        # 缓存rope_delta TODO 可能后面还得改
        # self.backbone.rope_deltas = mrope_position_deltas
        # print(input_embeddings.shape)
        # sys.stdout.flush()
        if labels_ids is not None:
            result: Qwen3VLCausalLMOutputWithPast = self.backbone(
                inputs_embeds=input_embeddings, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                labels=labels_ids
            )
            return result
        else:
            # 生成模式：使用 generate 方法生成序列
            result = self.backbone.generate(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=512,  # 可根据需要调整
                do_sample=False,  # 使用贪心解码
            )
            
        return result
