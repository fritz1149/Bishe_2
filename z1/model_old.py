from typing import Any, Iterator, List, Optional, Tuple, Union
from torch.nn import Parameter
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
import sys

def get_llm(args):
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.llm, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(f"./{args.llm}")
    return model

class ProposeModel(nn.Module, GenerationMixin):
    _is_stateful = Qwen3VLForConditionalGeneration._is_stateful
    def __init__(self, args):
        super().__init__()
        backbone = get_llm(args)
        backbone.model.visual = torch.nn.Identity()
        self.text_embedder = backbone.model.get_input_embeddings()
        self.generation_config = backbone.generation_config
        self.config = backbone.config
        self.main_input_name = backbone.main_input_name
        self.device = None
        
        if not args.test_mode:
            from peft import LoraConfig, PeftMixedModel, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            backbone = PeftMixedModel(backbone, lora_config, adapter_name="0")
        if args.align1_mode or args.align2_mode or args.finetune_mode or args.test_mode:
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
        elif args.test_mode:
            self.backbone = backbone
            for param in self.parameters(recurse=True):
                param.requires_grad = False
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
        # 如果self具有encoder这个成员
        if hasattr(self, "encoder"):
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
                # print(payload_ids.shape)
                # sys.stdout.flush()
                payload_embeddings = self.encoder((payload_ids, attention_mask_payload, global_attention_mask_payload))
                payload_embeddings = payload_embeddings.squeeze(1).to(text_device)
                # print(payload_embeddings.shape)
                input_payload_pos = input_ids[i] == IMAGE_PAD_ID
                # print(input_ids[i].shape, input_payload_pos.shape, input_payload_pos.sum().item())
                assert payload_embeddings.shape[0] == input_payload_pos.sum().item()
                inputs_embeds[i, input_payload_pos] = payload_embeddings
        
        if position_ids is None:
            assert cache_position is not None
            if rope_deltas is None:
                rope_deltas = 0
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (cache_position[0] + rope_deltas).to(inputs_embeds.device)
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)
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
        result: Qwen3VLCausalLMOutputWithPast = self.backbone(
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
        return result

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        if cache_position[0] != 0:
            model_inputs["position_ids"] = None
            model_inputs["payloads"] = None
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        assert expand_size == 1
        return input_ids, model_kwargs
            