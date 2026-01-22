import torch
from torch import nn
from typing import Iterator, List, Optional, Tuple, Union
from torch.nn import Parameter
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from z2.qwen3_vl_embedding import Qwen3VLForEmbeddingOutput, Qwen3VLForEmbedding

def get_embedder(args):
    from z2.qwen3_vl_embedding import Qwen3VLForEmbedding
    model = Qwen3VLForEmbedding.from_pretrained(args.llm, trust_remote_code=True)
    return model

class TrafficEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        backbone = get_embedder(args)
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
        self.backbone = PeftMixedModel(backbone, lora_config, adapter_name="0")
        
        from z1.encoder import get_longformer_with_projector
        self.encoder = get_longformer_with_projector(args)
        
        if args.train_mode:
            for name, param in self.named_parameters(recurse=True):
                if ("lora" not in name and "predictor" not in name and "fc" not in name) or "backbone.encoder.original_model" in name or not param.requires_grad:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        if args.eval_mode:
            for param in self.parameters(recurse=True):
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

    def resume(self, args):
        # payload encoder 从点1的训练结果中读取
        if getattr(args, "resume_encoder", None) and args.resume_encoder != "":
            ckpt = torch.load(args.resume_encoder, map_location="cpu", weights_only=True)
            # 仅加载 "encoder." 开头的权重
            state_dict = ckpt["model"]
            encoder_state_dict = {k[len("module.backbone.original_model."):]: v for k, v in state_dict.items() if k.startswith("module.backbone.original_model.")}
            incompatible_keys = self.encoder.original_model.load_state_dict(encoder_state_dict, strict=False)
            if getattr(args, "resume_log", False):
                print("resume encoder")
                for k in encoder_state_dict.keys():
                    print("module.backbone.original_model. "+k)
                print("Missing keys (模型有，但 checkpoint 没有):")
                print(incompatible_keys.missing_keys)
                print("\nUnexpected keys (checkpoint 有，但模型没有):")
                print(incompatible_keys.unexpected_keys)
        # payload encoder fc 从点2的训练结果中读取
        if getattr(args, "resume_linear", None) and args.resume_linear != "":
            ckpt = torch.load(args.resume_linear, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"]
            linear_state_dict = {k[len("module.backbone.encoder.fc."):]: v for k, v in state_dict.items() if k.startswith("module.backbone.encoder.fc.")}
            linear_state_dict.update({k[len("backbone.encoder.fc."):]: v for k, v in state_dict.items() if k.startswith("backbone.encoder.fc.")})
            incompatible_keys = self.encoder.fc.load_state_dict(linear_state_dict, strict=False)
            if getattr(args, "resume_log", False):
                print("resume linear")
                for k in linear_state_dict.keys():
                    print("module.backbone.encoder.fc. "+k)
                print("Missing keys (模型有，但 checkpoint 没有):")
                print(incompatible_keys.missing_keys)
                print("\nUnexpected keys (checkpoint 有，但模型没有):")
                print(incompatible_keys.unexpected_keys)
        # lora0 从点2的训练结果中读取
        if getattr(args, "resume_lora0", None) and args.resume_lora0 != "":
            ckpt = torch.load(args.resume_lora0, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"]
            lora0_state_dict = {k[len("module.backbone.backbone.base_model.model.model.language_model.layers."):]: v for k, v in state_dict.items() if k.startswith("module.backbone.backbone.base_model.model.model.language_model.layers.")}
            lora0_state_dict.update({k[len("backbone.backbone.base_model.model.model.language_model.layers."):]: v for k, v in state_dict.items() if k.startswith("backbone.backbone.base_model.model.model.language_model.layers.")})
            incompatible_keys = self.backbone.base_model.model.model.language_model.layers.load_state_dict(lora0_state_dict, strict=False)
            if getattr(args, "resume_log", False):
                print("resume lora0")
                for k in lora0_state_dict.keys():
                    print("module.backbone.backbone.base_model.model.model.language_model.layers. "+k)
                print("Missing keys (模型有，但 checkpoint 没有):")
                print(incompatible_keys.missing_keys)
                print("\nUnexpected keys (checkpoint 有，但模型没有):")
                print(incompatible_keys.unexpected_keys)

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            payloads: Optional[List[Tuple]] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[TransformersKwargs]
        ):
        IMAGE_PAD_ID = 151655
        text_device = next(self.text_embedder.parameters()).device
        encoder_device = next(self.encoder.parameters()).device

        input_ids = input_ids.to(text_device)
        assert inputs_embeds is None
        inputs_embeds = self.text_embedder(input_ids)
        # Clone inputs_embeds to avoid in-place operation on leaf variable that requires grad
        inputs_embeds = inputs_embeds.clone()

        if payloads is not None:
            for i, payload_tuples in enumerate(payloads):
                if payload_tuples is None:
                    input_payload_pos = input_ids[i] == IMAGE_PAD_ID
                    assert input_payload_pos.sum().item() == 0
                    continue
                payload_ids, attention_mask_payload, global_attention_mask_payload = payload_tuples
                payload_ids = payload_ids.to(encoder_device)
                attention_mask_payload = attention_mask_payload.to(encoder_device)
                global_attention_mask_payload = global_attention_mask_payload.to(encoder_device)
                payload_embeddings = self.encoder((payload_ids, attention_mask_payload, global_attention_mask_payload))
                payload_embeddings = payload_embeddings.squeeze(1).to(text_device, dtype=inputs_embeds.dtype)
                input_payload_pos = input_ids[i] == IMAGE_PAD_ID
                assert payload_embeddings.shape[0] == input_payload_pos.sum().item()
                inputs_embeds[i, input_payload_pos] = payload_embeddings
        
        # print(position_ids)

        # print("inputs_embeds.shape: ", inputs_embeds.shape)
        # print("position_ids.shape: ", position_ids.shape)
        # print("attention_mask.shape: ", attention_mask.shape)
        # print("position_ids: ")
        # for i in range(3):
        #     for j in range(position_ids.shape[2]):
        #         print(position_ids[i][0][j].item(), end=" ")
        #     print()
        # print()
        # sys.stdout.flush()
        attention_mask = attention_mask.to(text_device)
        position_ids = position_ids.to(text_device)
        result: Qwen3VLForEmbeddingOutput = self.backbone(
            input_ids=None,
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            cache_position=cache_position,
            labels=labels,
            logits_to_keep=logits_to_keep,
            **kwargs
        )
        embeddings = self._pooling_last(result.last_hidden_state, result.attention_mask)
        if kwargs.get('normalize', True):
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings