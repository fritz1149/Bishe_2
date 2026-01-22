"""Collate functions for various dataset types."""

import torch
from typing import List, Any, Dict


def collate_ContrastiveDataset(batch, args):
    x0 = [item[0][0][0] for item in batch]
    x0_mask_len = [item[0][0][1] for item in batch]
    x1 = [item[0][1][0] for item in batch]
    x1_mask_len = [item[0][1][1] for item in batch]
    y = [1 if item[1] == 1 else args.weak_sample_rate for item in batch]

    return torch.tensor(x0), torch.tensor(x0_mask_len), torch.tensor(x1), torch.tensor(x1_mask_len), torch.tensor(y)


def collate_ContrastiveDataset_test(batch):
    x = [item[0][0] for item in batch]
    x_mask_len = [item[0][1] for item in batch]
    y = [item[1] for item in batch]

    return torch.tensor(x), torch.tensor(x_mask_len), torch.tensor(y)


def reconstruct_payload(payloads):
    for i, item in enumerate(payloads):
        if len(item) == 0:
            payloads[i] = None
            continue
        payload_ids = torch.tensor([x[0] for x in item])
        attention_mask = torch.tensor([x[1] for x in item])
        global_attention_mask = torch.tensor([x[2] for x in item])
        payloads[i] = (payload_ids, attention_mask, global_attention_mask)
    return payloads


def reconstruct_input_ids(x_ids_list, max_seq_len, PAD_ID):
    input_ids = []
    for x_ids in x_ids_list:
        input_seq = x_ids
        # 补齐
        pad_len = max_seq_len - len(input_seq)
        input_seq += [PAD_ID] * pad_len
        input_ids.append(input_seq)
    return torch.tensor(input_ids)


def construct_attention_mask(input_ids, PAD_ID):
    attention_mask = (input_ids != PAD_ID).long()
    for i in range(attention_mask.size(0)):
        zero_idx = (attention_mask[i] == 0).nonzero(as_tuple=True)[0]
        if len(zero_idx) > 0:
            attention_mask[i, zero_idx[0]] = 1
    return attention_mask


def reconstruct_position_ids(position_ids_list, max_seq_len):
    position_ids = []
    for position_ids_ in position_ids_list:
        start = position_ids_.max().item()+1
        position_ids.append(torch.cat([position_ids_, torch.arange(start, start+max_seq_len-position_ids_.shape[1]).unsqueeze(0).expand(3, -1)], dim=1))
    return torch.stack(position_ids, dim=1)


def collate_ContrastiveDataset2(batch):
    PAD_ID = 151643
    IMAGE_PAD_ID = 151655
    x_ids_1 = [item[0][0][0] for item in batch]
    payloads_1 = [item[0][0][2] for item in batch] # [batch_size, row_num_sample, (1500, 1500, 1500)]
    position_ids_1 = [item[0][0][3] for item in batch] # [batch_size, 3, total_seq_len_sample_1+total_seq_len_sample_2]
    x_ids_2 = [item[0][1][0] for item in batch]
    payloads_2 = [item[0][1][2] for item in batch] # [batch_size, row_num_sample, (1500, 1500, 1500)]
    position_ids_2 = [item[0][1][3] for item in batch] # [batch_size, 3, total_seq_len_sample_1+total_seq_len_sample_2]

    payloads_1 = reconstruct_payload(payloads_1)
    payloads_2 = reconstruct_payload(payloads_2)

    # 计算每个样本的总长度
    seq_lens = [
        len(x_ids) for x_ids in x_ids_1 + x_ids_2
    ]
    max_seq_len = max(seq_lens)

    input_ids_1 = reconstruct_input_ids(x_ids_1, max_seq_len, PAD_ID)
    input_ids_2 = reconstruct_input_ids(x_ids_2, max_seq_len, PAD_ID)

    # payload_ids = torch.tensor(payload_ids) # payload这里每个样本的序列长度不一样
    position_ids_1 = reconstruct_position_ids(position_ids_1, max_seq_len)
    position_ids_2 = reconstruct_position_ids(position_ids_2, max_seq_len)

    
    attention_mask_1 = construct_attention_mask(input_ids_1, PAD_ID)
    attention_mask_2 = construct_attention_mask(input_ids_2, PAD_ID)

    assert position_ids_1.shape[0] == 3 
    assert position_ids_1.shape[1] == input_ids_1.shape[0]
    assert position_ids_1.shape[2] == max_seq_len and position_ids_1.shape[2] == input_ids_1.shape[1]
    assert position_ids_2.shape[0] == 3 
    assert position_ids_2.shape[1] == input_ids_2.shape[0]
    assert position_ids_2.shape[2] == max_seq_len and position_ids_2.shape[2] == input_ids_2.shape[1]

    # rope_deltas_1 = position_ids_1.max().item()+1-input_ids_1.shape[1] # [batch_size]
    # rope_deltas_2 = position_ids_2.max().item()+1-input_ids_2.shape[1] # [batch_size]
        
    out_1 = {
        "input_ids": input_ids_1,
        "payloads": payloads_1,
        "position_ids": position_ids_1,
        "attention_mask": attention_mask_1
    }
    out_2 = {
        "input_ids": input_ids_2,
        "payloads": payloads_2,
        "position_ids": position_ids_2,
        "attention_mask": attention_mask_2
    }
    return out_1, out_2


def collate_ContrastiveDataset_test2(batch):
    PAD_ID = 151643
    IMAGE_PAD_ID = 151655
    x_ids = [item[0][0] for item in batch]
    payloads = [item[0][2] for item in batch] # [batch_size, row_num_sample, (1500, 1500, 1500)]
    position_ids = [item[0][3] for item in batch] # [batch_size, 3, total_seq_len_sample_1+total_seq_len_sample_2]
    y = [item[1] if item[1] is not None else 0 for item in batch]

    seq_lens = [
        len(x) for x in x_ids
    ]
    max_seq_len = max(seq_lens)
    input_ids = reconstruct_input_ids(x_ids, max_seq_len, PAD_ID)
    payloads = reconstruct_payload(payloads)
    position_ids = reconstruct_position_ids(position_ids, max_seq_len)
    attention_mask = construct_attention_mask(input_ids, PAD_ID)
    assert position_ids.shape[0] == 3 
    assert position_ids.shape[1] == input_ids.shape[0]
    assert position_ids.shape[2] == max_seq_len and position_ids.shape[2] == input_ids.shape[1]

    # rope_deltas = position_ids.max().item()+1-input_ids.shape[1] # [batch_size]

    data = {
        "input_ids": input_ids,
        "payloads": payloads,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }
    return data, torch.tensor(y)


def collate_TrafficEmbeddingDataset(batch):
    PAD_ID = 151643
    IMAGE_PAD_ID = 151655
    x_ids = [item[0][0] for item in batch]
    payloads = [item[0][2] for item in batch] # [batch_size, row_num_sample, (1500, 1500, 1500)]
    position_ids = [item[0][3] for item in batch] # [batch_size, 3, total_seq_len_sample_1+total_seq_len_sample_2]
    y = [item[1] for item in batch]

    seq_lens = [
        len(x) for x in x_ids
    ]
    max_seq_len = max(seq_lens)
    input_ids = reconstruct_input_ids(x_ids, max_seq_len, PAD_ID)
    payloads = reconstruct_payload(payloads)
    position_ids = reconstruct_position_ids(position_ids, max_seq_len)
    attention_mask = construct_attention_mask(input_ids, PAD_ID)
    assert position_ids.shape[0] == 3 
    assert position_ids.shape[1] == input_ids.shape[0]
    assert position_ids.shape[2] == max_seq_len and position_ids.shape[2] == input_ids.shape[1]

    # rope_deltas = position_ids.max().item()+1-input_ids.shape[1] # [batch_size]

    data = {
        "input_ids": input_ids,
        "payloads": payloads,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }
    return data, y


def collate_LLMDataset(batch):
    PAD_ID = 151643
    IMAGE_PAD_ID = 151655
    x_ids = [item[0][0] for item in batch]
    y_ids = [item[0][1] for item in batch] # [batch_size, seq_len_sample_1]
    payloads = [item[0][2] for item in batch] # [batch_size, row_num_sample, (1500, 1500, 1500)]
    position_ids = [item[0][3] for item in batch] # [batch_size, 3, total_seq_len_sample_1+total_seq_len_sample_2]
    labels = [item[1] for item in batch] # [batch_size]
    # import sys
    # # print("x_ids_len:", len(x_ids[0]), "y_ids_len:", len(y_ids[0]))
    # sys.stdout.flush()

    for i, item in enumerate(payloads):
        if len(item) == 0:
            payloads[i] = None
            continue
        payload_ids = torch.tensor([x[0] for x in item])
        attention_mask = torch.tensor([x[1] for x in item])
        global_attention_mask = torch.tensor([x[2] for x in item])
        payloads[i] = (payload_ids, attention_mask, global_attention_mask)

    # 计算每个样本的总长度
    seq_lens = [
        len(x_ids) + len(y_ids)
        for x_ids, y_ids in zip(x_ids, y_ids)
    ]
    max_seq_len = max(seq_lens)

    input_ids = []
    target_labels = []
    for x_ids, y_ids in zip(x_ids, y_ids):
        input_seq = x_ids+y_ids
        # 补齐
        pad_len = max_seq_len - len(input_seq)
        input_seq += [PAD_ID] * pad_len
        input_ids.append(input_seq)

        # 构造label，非label部分-100
        label_prefix = [-100] * len(x_ids)
        label_suffix = [-100] * pad_len
        target_labels.append(label_prefix + y_ids + label_suffix)

    input_ids = torch.tensor(input_ids)
    labels_ids = torch.tensor(target_labels)
    # payload_ids = torch.tensor(payload_ids) # payload这里每个样本的序列长度不一样
    for i, position_ids_ in enumerate(position_ids):
        start = position_ids_.max().item()+1
        position_ids[i] = torch.cat([position_ids_, torch.arange(start, start+max_seq_len-position_ids_.shape[1]).unsqueeze(0).expand(3, -1)], dim=1)
    position_ids = torch.stack(position_ids, dim=1)
    attention_mask = (input_ids != PAD_ID).long()
    assert position_ids.shape[0] == 3 
    assert position_ids.shape[1] == input_ids.shape[0]
    assert position_ids.shape[2] == max_seq_len and position_ids.shape[2] == input_ids.shape[1]

    # rope_deltas = position_ids.max().item()+1-input_ids.shape[1] # [batch_size]
        
    return input_ids, labels_ids, payloads, position_ids, attention_mask, labels


def collate_LLMDataset_leftpadding(batch, keep_labels=False):
    PAD_ID = 151643
    IMAGE_PAD_ID = 151655
    x_ids = [item[0][0] for item in batch]
    y_ids = [item[0][1] for item in batch] # [batch_size, seq_len_sample_1]
    payloads = [item[0][2] for item in batch] # [batch_size, row_num_sample, (1500, 1500, 1500)]
    position_ids = [item[0][3] for item in batch] # [batch_size, 3, total_seq_len_sample_1+total_seq_len_sample_2]
    labels = [item[1] for item in batch] # [batch_size]
    # import sys
    # # print("x_ids_len:", len(x_ids[0]), "y_ids_len:", len(y_ids[0]))
    # sys.stdout.flush()

    for i, item in enumerate(payloads):
        if len(item) == 0:
            payloads[i] = None
            continue
        payload_ids = torch.tensor([x[0] for x in item])
        attention_mask = torch.tensor([x[1] for x in item])
        global_attention_mask = torch.tensor([x[2] for x in item])
        payloads[i] = (payload_ids, attention_mask, global_attention_mask)

    # 计算每个样本的总长度
    seq_lens = [
        len(x_ids) + len(y_ids)
        for x_ids, y_ids in zip(x_ids, y_ids)
    ]
    max_seq_len = max(seq_lens)

    input_ids = []
    target_labels = []
    for x_ids, y_ids in zip(x_ids, y_ids):
        input_seq = x_ids+y_ids
        # 补齐
        pad_len = max_seq_len - len(input_seq)
        input_seq = [PAD_ID] * pad_len + input_seq
        input_ids.append(input_seq)

        # 构造label，非label部分-100
        if keep_labels:
            label_prefix = [-100] * len(x_ids)
            label_suffix = [-100] * pad_len
            target_labels.append(label_prefix + y_ids + label_suffix)

    input_ids = torch.tensor(input_ids)
    if keep_labels:
        labels_ids = torch.tensor(target_labels)
    # payload_ids = torch.tensor(payload_ids) # payload这里每个样本的序列长度不一样
    for i, position_ids_ in enumerate(position_ids):
        start = position_ids_.max().item()+1
        position_ids[i] = torch.cat([position_ids_, torch.arange(start, start+max_seq_len-position_ids_.shape[1]).unsqueeze(0).expand(3, -1)], dim=1)
    position_ids = torch.stack(position_ids, dim=1)
    attention_mask = (input_ids != PAD_ID).long()
    assert position_ids.shape[0] == 3 
    assert position_ids.shape[1] == input_ids.shape[0]
    assert position_ids.shape[2] == max_seq_len and position_ids.shape[2] == input_ids.shape[1]

    # rope_deltas = position_ids.max().item()+1-input_ids.shape[1] # [batch_size]
        
    return {
        "input_ids": input_ids,
        "labels": labels_ids if keep_labels else None,
        "payloads": payloads,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }, labels
