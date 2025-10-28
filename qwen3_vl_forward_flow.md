# Qwen3VL 模型 Forward 过程梳理

## 模型结构概览

Qwen3VL 模型主要由三个核心组件组成：
1. **Qwen3VLVisionModel**: 视觉编码器，处理图像和视频
2. **Qwen3VLTextModel**: 文本编码器（基于Qwen3的Transformer）
3. **Qwen3VLModel**: 多模态融合模型，协调视觉和文本处理

## Forward 流程详解

### 主入口：`Qwen3VLModel.forward()` (line 1108-1239)

#### 1. 输入准备阶段 (line 1128-1132)
```python
# 输入验证
if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

# 获取文本嵌入
if inputs_embeds is None:
    inputs_embeds = self.get_input_embeddings()(input_ids)
```
- 确保 `input_ids` 或 `inputs_embeds` 二选一
- 如果只有 `input_ids`，通过词嵌入层转换为 `inputs_embeds`

#### 2. 图像特征提取 (line 1137-1143)
```python
if pixel_values is not None:
    # 获取图像特征
    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    # 获取图像占位符mask
    image_mask, _ = self.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    # 用图像特征替换占位符
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

**`get_image_features()` 流程** (line 1050-1064):
1. 调用 `self.visual(pixel_values, grid_thw=image_grid_thw)` 进行视觉编码
2. 根据 `image_grid_thw` 分割特征
3. 返回图像嵌入和 deepstack 特征（多层级特征）

**`Qwen3VLVisionModel.forward()` 流程** (line 703-753):
1. **Patch Embedding** (line 714): `self.patch_embed(hidden_states)`
   - 使用 3D 卷积将图像/视频切分为 patches
   - 输出: `(seq_len, hidden_size)`

2. **位置编码** (line 716-725):
   - **固定位置编码**: `self.fast_pos_embed_interpolate(grid_thw)` - 双线性插值
   - **旋转位置编码**: `self.rot_pos_emb(grid_thw)` - 用于空间位置的RoPE
   - 两者相加得到最终位置编码

3. **Vision Transformer Blocks** (line 738-749):
   ```python
   for layer_num, blk in enumerate(self.blocks):
       hidden_states = blk(hidden_states, cu_seqlens, position_embeddings, ...)
       # 收集 deepstack 特征（从指定层）
       if layer_num in self.deepstack_visual_indexes:
           deepstack_feature = self.deepstack_merger_list[...](hidden_states)
           deepstack_feature_lists.append(deepstack_feature)
   ```
   - 通过多个 Vision Block 处理（包含 Self-Attention + MLP）
   - 在指定层提取 deepstack 特征用于后续融合

4. **Patch Merger** (line 751): `self.merger(hidden_states)`
   - 将多个空间 patch 合并，减少序列长度
   - 返回最终视觉特征和 deepstack 特征列表

#### 3. 视频特征提取 (line 1145-1151)
```python
if pixel_values_videos is not None:
    video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
    video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    _, video_mask = self.get_placeholder_mask(...)
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
```
- 与图像处理类似，但 `get_video_features()` 内部调用 `get_image_features()`
- 视频被处理为多帧图像序列

#### 4. 视觉特征融合 (line 1153-1175)
```python
# 合并图像和视频的mask和deepstack特征
if image_mask is not None and video_mask is not None:
    visual_pos_masks = image_mask | video_mask
    # 聚合 deepstack 特征
    deepstack_visual_embeds = [...]
```
- 将图像和视频的位置mask合并为 `visual_pos_masks`
- 聚合图像和视频的 deepstack 特征

#### 5. 位置ID计算 (line 1177-1221)
```python
if position_ids is None:
    # 计算 RoPE 位置索引
    position_ids, rope_deltas = self.get_rope_index(
        input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask_tensor
    )
    self.rope_deltas = rope_deltas
```

**`get_rope_index()` 关键逻辑** (line 916-1033):
- **多模态RoPE (mRoPE)**: 为文本、图像、视频计算不同的位置编码
- **3D位置编码**: `position_ids` 形状为 `(3, batch_size, seq_len)`，分别对应：
  - `position_ids[0]`: 时间维度（对视频）
  - `position_ids[1]`: 高度维度
  - `position_ids[2]`: 宽度维度
- **Delta计算**: 计算 `rope_deltas` 用于推理时的位置增量更新

#### 6. 语言模型前向传播 (line 1223-1233)
```python
outputs = self.language_model(
    input_ids=None,  # 已转换为 inputs_embeds
    position_ids=position_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,  # 已融合视觉特征
    cache_position=cache_position,
    visual_pos_masks=visual_pos_masks,  # 视觉位置mask
    deepstack_visual_embeds=deepstack_visual_embeds,  # DeepStack特征
    **kwargs,
)
```

**`Qwen3VLTextModel.forward()` 流程** (line 784-874):

1. **输入嵌入** (line 813-814):
   ```python
   if inputs_embeds is None:
       inputs_embeds = self.embed_tokens(input_ids)
   ```

2. **位置编码处理** (line 823-833):
   - 处理3D位置ID（时间、高度、宽度）
   - 提取文本位置ID用于注意力计算

3. **创建因果掩码** (line 834-841):
   ```python
   attention_mask = create_causal_mask(...)
   ```

4. **生成RoPE位置嵌入** (line 846):
   ```python
   position_embeddings = self.rotary_emb(hidden_states, position_ids)
   ```

5. **Decoder Layers 处理** (line 849-868):
   ```python
   for layer_idx, decoder_layer in enumerate(self.layers):
       # Transformer解码层
       layer_outputs = decoder_layer(
           hidden_states,
           attention_mask=attention_mask,
           position_ids=text_position_ids,
           past_key_values=past_key_values,
           cache_position=cache_position,
           position_embeddings=position_embeddings,
           **kwargs,
       )
       hidden_states = layer_outputs
       
       # DeepStack: 将视觉特征注入到早期层
       if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
           hidden_states = self._deepstack_process(
               hidden_states,
               visual_pos_masks,
               deepstack_visual_embeds[layer_idx],
           )
   ```
   - 每个解码层包含：
     - Self-Attention（使用RoPE位置编码）
     - MLP
   - **DeepStack机制**: 在前几层将视觉特征直接加到对应的隐藏状态上

6. **层归一化** (line 869):
   ```python
   hidden_states = self.norm(hidden_states)
   ```

7. **DeepStack处理函数** (line 876-883):
   ```python
   def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
       # 在视觉位置处添加视觉特征
       local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
       hidden_states[visual_pos_masks, :] = local_this
       return hidden_states
   ```

#### 7. 输出构建 (line 1235-1239)
```python
return Qwen3VLModelOutputWithPast(
    last_hidden_state=outputs.last_hidden_state,
    past_key_values=outputs.past_key_values,
    rope_deltas=self.rope_deltas,
)
```

## 关键特性总结

### 1. **多模态融合方式**
- **占位符替换**: 图像/视频特征直接替换输入序列中的占位符token
- **DeepStack机制**: 在文本编码器的早期层注入多层级视觉特征，增强视觉-语言对齐

### 2. **位置编码机制**
- **mRoPE (Multi-modal RoPE)**: 
  - 文本使用1D位置编码
  - 图像/视频使用3D位置编码（时间、高度、宽度）
  - 支持不同模态的混合序列

### 3. **视觉编码流程**
```
输入图像/视频 
→ Patch Embedding (3D Conv)
→ 位置编码 (固定 + RoPE)
→ Vision Transformer Blocks (多层级处理)
→ Patch Merger (降维)
→ 输出特征 + DeepStack特征
```

### 4. **文本编码流程**
```
文本嵌入 + 视觉特征嵌入
→ 位置编码 (mRoPE)
→ Decoder Layers:
  - Self-Attention
  - MLP
  - DeepStack特征注入（前几层）
→ Layer Norm
→ 输出隐藏状态
```

### 5. **推理优化**
- 缓存 `rope_deltas` 避免重复计算
- 支持 KV cache (`past_key_values`)
- 支持 Flash Attention 加速

## 数据流示意图

```
输入:
├── input_ids/pixel_values/pixel_values_videos
└── attention_mask/position_ids/...

处理流程:
├── [文本] input_ids → inputs_embeds
├── [图像] pixel_values → Qwen3VLVisionModel → image_embeds + deepstack_image_embeds
├── [视频] pixel_values_videos → Qwen3VLVisionModel → video_embeds + deepstack_video_embeds
├── [融合] masked_scatter(占位符位置替换为视觉特征)
├── [位置] get_rope_index() → position_ids (3D) + rope_deltas
└── [语言模型] Qwen3VLTextModel(
        inputs_embeds (融合后的),
        position_ids,
        deepstack_visual_embeds,
        visual_pos_masks,
        ...
    )

输出:
└── Qwen3VLModelOutputWithPast(
        last_hidden_state,
        past_key_values,
        rope_deltas
    )
```

