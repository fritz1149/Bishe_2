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
from types import SimpleNamespace
import sys

def get_llm(args):
    torch_dtype = getattr(args, 'torch_dtype', None)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.llm, trust_remote_code=True,
        torch_dtype=torch_dtype
    )
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
        
        from peft import LoraConfig, PeftMixedModel, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        backbone = PeftMixedModel(backbone, lora_config, adapter_name="0")
        from .encoder import get_longformer_with_projector
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
            if args.wo_weight_mode or args.tllm_mode:
                backbone.set_adapter(["1"])
            else:
                backbone.set_adapter(["0", "1"])
            self.backbone = backbone
            # # 冻结除lora参数以外的所有参数
            for name, param in self.named_parameters(recurse=True):
                if "lora_A.1" not in name and "lora_B.1" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                if args.wo_weight_mode and "encoder" in name:
                    param.requires_grad = True
        elif args.test_mode:
            self.backbone = backbone
            for param in self.parameters(recurse=True):
                param.requires_grad = False
        elif args.eval_mode:
            backbone.add_adapter("1", lora_config)
            if args.wo_weight_mode or args.tllm_mode:
                backbone.set_adapter(["1"])
            else:
                backbone.set_adapter(["0", "1"])
            self.backbone = backbone
            for param in self.parameters(recurse=True):
                param.requires_grad = False
        if args.z2_mode:
            self.backbone = backbone
            label_num = len(args.labels)
            self.label2id = {label: idx for idx, label in enumerate(sorted(args.labels))}
            hidden_size = args.linear_output_dim
            self.classifier = nn.Linear(hidden_size, label_num)
            import torch.nn.init as init
            init.normal_(self.classifier.weight, std=0.01)
            init.zeros_(self.classifier.bias)
            for param in self.classifier.parameters():
                param.requires_grad = True
            
            # momentum k-encoder 支持
            self.momentum_k = getattr(args, 'momentum_k', 0.0)
            if self.momentum_k > 0:
                self._init_k_encoder(args)
        # for name, param in self.named_parameters(recurse=True):
        #     if param.requires_grad:
        #         print(name)
        # for name, module in self.named_modules():
        #     print(name, type(module).__name__)
        # for name, param in self.named_parameters(recurse=True):
        #     print(name)

    def _init_k_encoder(self, args):
        """初始化 momentum k-encoder，仅保存可学习参数的副本（不创建完整模型副本）"""
        # 保存可学习参数的副本，用于动量更新
        # 格式: {param_name: param_data.clone()}
        self.k_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.k_params[name] = param.data.clone()
    
    @torch.no_grad()
    def momentum_update_k_encoder(self):
        """使用动量更新 k-encoder 的参数"""
        if not hasattr(self, 'k_params'):
            return
        m = self.momentum_k
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.k_params:
                self.k_params[name] = self.k_params[name] * m + param.data * (1. - m)

    @torch.no_grad()
    def forward_k(self, input_ids, attention_mask, position_ids, payloads, classifier_labels=None):
        """使用 k-encoder 进行前向传播，通过动态替换权重实现
        
        步骤：
        1. 保存当前可学习参数
        2. 加载 k-encoder 参数
        3. 执行前向传播
        4. 恢复原参数
        """
        if not hasattr(self, 'k_params'):
            raise RuntimeError("k-encoder not initialized. Set momentum_k > 0 to enable.")
        
        # 1. 保存当前可学习参数
        original_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.k_params:
                original_params[name] = param.data.clone()
        
        # 2. 加载 k-encoder 参数
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.k_params:
                param.data.copy_(self.k_params[name])
        
        try:
            # 3. 执行前向传播（复用主模型的 forward）
            result = self.forward(
                input_ids=input_ids,
                labels=None,
                payloads=payloads,
                position_ids=position_ids,
                attention_mask=attention_mask,
                classifier_labels=classifier_labels,
                rope_deltas=None
            )
            
            output = SimpleNamespace(
                last_hidden_states=result.last_hidden_states.detach(),
                logits=result.logits.detach() if result.logits is not None else None
            )
        except Exception as e:
            print(f"Error in forward_k: {e}")
            raise
        finally:
            # 4. 恢复原参数
            for name, param in self.named_parameters():
                if param.requires_grad and name in original_params:
                    param.data.copy_(original_params[name])
        
        return output

    def parameters_(self, recurse: bool = True) -> Iterator[Parameter]:
        return [p for p in self.parameters(recurse=recurse) if p.requires_grad]
    def named_parameters_names(self, recurse: bool = True) -> Iterator[str]:
        return [name for name, p in self.named_parameters(recurse=recurse) if p.requires_grad]
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

    def dispatch(self, device_map=None, split_layers_num=25, single_gpu=False):
        self.device = torch.device('cuda:0')
        self.encoder = self.encoder.to('cuda:0')
        self.text_embedder = self.text_embedder.to('cuda:0')
        if hasattr(self, 'classifier'):
            self.classifier = self.classifier.to('cuda:1')
        if device_map is None:
            device_map = {
                "base_model.model.model.language_model.embed_tokens": "cuda:0",
                "base_model.model.model.language_model.rotary_emb": "cuda:0",
                "base_model.model.model.visual": "cuda:0",
                "base_model.model.lm_head": "cuda:1",
                "base_model.model.model.language_model.norm": "cuda:1",
                **{f"base_model.model.model.language_model.layers.{i}": "cuda:0" for i in range(0, split_layers_num)},
                **{f"base_model.model.model.language_model.layers.{i}": "cuda:1" for i in range(split_layers_num, 36)},
            }
        from accelerate import dispatch_model
        self.backbone = dispatch_model(
            self.backbone, 
            device_map=device_map, 
            main_device="cuda:1" if hasattr(self, 'classifier') else "cpu",
            skip_keys="hidden_states"  # 防止 hidden_states 被移动到输入设备
        )

    def resume(self, args):
        def resume_training_status(args, ckpt):
            if 'epoch' in ckpt:
                args.start_epoch = ckpt['epoch']+1
            # if 'optimizer' in ckpt:
            #     args.optimizer.load_state_dict(ckpt['optimizer'])
                # 否则忽略
            if 'best_loss' in ckpt:
                args.best_loss = ckpt['best_loss']
            if 'best_acc' in ckpt:
                args.best_acc = ckpt['best_acc']
            if 'scaler' in ckpt:
                args._scaler_state_dict = ckpt['scaler']
            else:
                args._scaler_state_dict = None

        if getattr(args, "resume_encoder", None) and args.resume_encoder != "":
            ckpt = torch.load(args.resume_encoder, map_location="cpu", weights_only=True)
            # 仅加载 "encoder." 开头的权重
            state_dict = ckpt["model"]
            if not args.wo_weight_mode:
                encoder_state_dict = {k[len("module.backbone.original_model."):]: v for k, v in state_dict.items() if k.startswith("module.backbone.original_model.")}
            else:
                encoder_state_dict = {k[len("encoder.original_model."):]: v for k, v in state_dict.items() if k.startswith("encoder.original_model.")}
            incompatible_keys = self.encoder.original_model.load_state_dict(encoder_state_dict, strict=False)
            if getattr(args, "resume_log", False):
                print("resume encoder")
                for k in encoder_state_dict.keys():
                    print("encoder.original_model. "+k)
                print("Missing keys (模型有，但 checkpoint 没有):")
                print(incompatible_keys.missing_keys)
                print("\nUnexpected keys (checkpoint 有，但模型没有):")
                print(incompatible_keys.unexpected_keys)
        if getattr(args, "resume_linear", None) and args.resume_linear != "":
            ckpt = torch.load(args.resume_linear, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"]
            linear_state_dict = {k[len("encoder.fc."):]: v for k, v in state_dict.items() if k.startswith("encoder.fc.")}
            incompatible_keys = self.encoder.fc.load_state_dict(linear_state_dict, strict=False)
            if getattr(args, "align2_mode", False):
                resume_training_status(args, ckpt)
            if getattr(args, "resume_log", False):
                print("resume linear")
                for k in linear_state_dict.keys():
                    print("encoder.fc. "+k)
                print("Missing keys (模型有，但 checkpoint 没有):")
                print(incompatible_keys.missing_keys)
                print("\nUnexpected keys (checkpoint 有，但模型没有):")
                print(incompatible_keys.unexpected_keys)
        if getattr(args, "resume_lora0", None) and args.resume_lora0 != "":
            ckpt = torch.load(args.resume_lora0, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"]
            lora0_state_dict = {k[len("backbone.base_model.model.model.language_model.layers."):]: v for k, v in state_dict.items() if k.startswith("backbone.base_model.model.model.language_model.layers.")}
            incompatible_keys = self.backbone.base_model.model.model.language_model.layers.load_state_dict(lora0_state_dict, strict=False)
            if getattr(args, "align1_mode", False):
                resume_training_status(args, ckpt)
            if getattr(args, "resume_log", False):
                print("resume lora0")
                for k in lora0_state_dict.keys():
                    print("backbone.base_model.model.model.language_model.layers. "+k)
                print("Missing keys (模型有，但 checkpoint 没有):")
                print(incompatible_keys.missing_keys)
                print("\nUnexpected keys (checkpoint 有，但模型没有):")
                print(incompatible_keys.unexpected_keys)
        if getattr(args, "resume_lora1", None) and args.resume_lora1 != "":
            ckpt = torch.load(args.resume_lora1, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"]
            lora1_state_dict = {k[len("backbone.base_model.model.model.language_model.layers."):]: v for k, v in state_dict.items() if k.startswith("backbone.base_model.model.model.language_model.layers.")}
            incompatible_keys = self.backbone.base_model.model.model.language_model.layers.load_state_dict(lora1_state_dict, strict=False)
            if getattr(args, "finetune_mode", False):
                resume_training_status(args, ckpt)
            if getattr(args, "resume_log", False):
                print("resume lora1")
                for k in state_dict.keys():
                    print(k)
                print("Missing keys (模型有，但 checkpoint 没有):")
                print(incompatible_keys.missing_keys)
                print("\nUnexpected keys (checkpoint 有，但模型没有):")
                print(incompatible_keys.unexpected_keys)

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
            classifier_labels: Optional[str] = None,
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
                payload_embeddings = payload_embeddings.squeeze(1).to(dtype=inputs_embeds.dtype, device=text_device)
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
            return_hidden_states=True if hasattr(self, "classifier") else False,
            **kwargs
        )

        if hasattr(self, "classifier"):
            # Get the last non-padding token's hidden state for each batch
            hidden_states = result.hidden_states  # (batch_size, seq_len, hidden_dim)

            batch_size = hidden_states.shape[0]
            
            # Find the last non-padding position for each batch
            last_hidden_states = []
            for i in range(batch_size):
                # Find positions that are not padding (assuming padding tokens have attention_mask == 0)
                assert attention_mask is not None
                non_pad_mask = attention_mask[i].bool()
                last_non_pad_idx = non_pad_mask.sum() - 1
                last_hidden_states.append(hidden_states[i, last_non_pad_idx])
            
            last_hidden_states = torch.stack(last_hidden_states, dim=0)  # (batch_size, hidden_dim)
            classifier_logits = self.classifier(last_hidden_states)
            result = SimpleNamespace(
                logits=classifier_logits,
                last_hidden_states=last_hidden_states,
                loss=None
            )
            
            # Calculate classification loss if labels are provided in kwargs
            if classifier_labels is not None:
                classifier_labels = torch.tensor([self.label2id[x] for x in classifier_labels], device=classifier_logits.device)
                classifier_loss = torch.nn.functional.cross_entropy(classifier_logits, classifier_labels)
                result.loss = classifier_loss
                        
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