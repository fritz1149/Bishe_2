def _bert_tokenizer(vocab_path: str = "config/encryptd_vocab.txt"):
    # 懒加载的tokenizer
    if not hasattr(_bert_tokenizer, "_cache"):
        import argparse
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        from uer.uer.opts import tokenizer_opts
        tokenizer_opts(parser)
        args = parser.parse_args([])  # 使用空列表，采用默认参数而非命令行
        args.vocab_path = vocab_path
        from uer.uer.utils import str2tokenizer
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        args.vocab = args.tokenizer.vocab
        SEQ_LENGTH = 1500
        from uer.uer.utils import CLS_TOKEN, PAD_TOKEN
        PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
        CLS_ID = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN])[0]
        _bert_tokenizer._cache = (args.tokenizer, SEQ_LENGTH, CLS_TOKEN, PAD_TOKEN, CLS_ID, PAD_ID)
    return _bert_tokenizer._cache

def _Qwen3VL_tokenizer(model_path: str = "./Qwen3-VL-8B-Instruct", SEQ_LENGTH: int = 1024):
    # 懒加载的tokenizer
    if not hasattr(_Qwen3VL_tokenizer, "_cache"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        PAD_TOKEN = tokenizer.special_tokens_map.get("pad_token", None)
        PAD_ID = tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
        _Qwen3VL_tokenizer._cache = (tokenizer, SEQ_LENGTH, None, PAD_TOKEN, None, PAD_ID)
    return _Qwen3VL_tokenizer._cache

def _cut_bursts(in_path):
    bursts = []
    with open(in_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        src_ip = lines[0][:-1].split("\t")[src_index]
        current_burst = []
        payloads_in_current_burst = []
        payloads_index_in_current_burst = []
        i = 0
        for line in lines:
            values = line[:-1].split("\t")
            ip = values[src_index]
            if src_ip != ip:
                bursts.append({"packets": current_burst, "payloads": payloads_in_current_burst, "payloads_index": payloads_index_in_current_burst})
                current_burst = []
                payloads_in_current_burst = []
                payloads_index_in_current_burst = []
                src_ip = ip
                i = 0
            current_burst.append(line)
            payload = values[tcp_payload_index]
            if payload == "":
                payload = values[udp_payload_index]
            if payload != "":
                payloads_in_current_burst.append(payload)
                payloads_index_in_current_burst.append(i)
            i += 1
        if len(current_burst) > 0:
                bursts.append({"packets": current_burst, "payloads": payloads_in_current_burst, "payloads_index": payloads_index_in_current_burst})
        return bursts

def _dump_in_chunks(items, out_dir, chunk_size, name=""):
    import os
    import pickle
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    if chunk_size == -1:
        chunk_size = len(items)
    assert len(items) > 0
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        file_name = f"{name}_part_{idx:05d}.pkl"
        file_path = os.path.join(out_dir, file_name)
        with open(file_path, "wb") as fout:
            pickle.dump(chunk, fout)
        idx += 1

fields = ["frame.encap_type", "frame.time", "frame.offset_shift", "frame.time_epoch", "frame.time_delta",
                "frame.time_relative", "frame.number", "frame.len", "frame.marked", "frame.protocols", "eth.dst",
                "eth.dst_resolved", "eth.src", "eth.src_resolved", "eth.type",
                "ip.version", "ip.hdr_len", "ip.dsfield", "ip.dsfield.dscp", "ip.dsfield.ecn", "ip.len", "ip.id",
                "ip.flags", "ip.flags.rb", "ip.flags.df", "ip.flags.mf", "ip.frag_offset", "ip.ttl", "ip.proto",
                "ip.checksum", "ip.checksum.status", "ip.src", "ip.dst", "tcp.srcport", "tcp.dstport", "tcp.stream",
                "tcp.len", "tcp.seq", "tcp.nxtseq", "tcp.ack", "tcp.hdr_len", "tcp.flags",
                "tcp.flags.res", "tcp.flags.cwr", "tcp.flags.urg", "tcp.flags.ack",
                "tcp.flags.push", "tcp.flags.reset", "tcp.flags.syn", "tcp.flags.fin", "tcp.flags.str",
                "tcp.window_size", "tcp.window_size_scalefactor", "tcp.checksum", "tcp.checksum.status", "tcp.urgent_pointer",
                "tcp.time_relative", "tcp.time_delta", "tcp.analysis.bytes_in_flight", "tcp.analysis.push_bytes_sent", "tcp.segment",
                "tcp.segment.count", "tcp.reassembled.length", "tcp.payload", "udp.srcport", "udp.dstport", "udp.length",
                "udp.checksum", "udp.checksum.status", "udp.stream", "data.len", "udp.payload"]

src_index = fields.index("ip.src")
dst_index = fields.index("ip.dst")
ip_version_index = fields.index("ip.version")
tcp_payload_index = fields.index("tcp.payload")
udp_payload_index = fields.index("udp.payload")
tcp_segment_index = fields.index("tcp.segment")
tcp_segment_count_index = fields.index("tcp.segment.count")
frame_protocol_index = fields.index("frame.protocols")
udp_srcport_index = fields.index("udp.srcport")
tcp_srcport_index = fields.index("tcp.srcport")
tcp_field_indexes = [i for i, field in enumerate(fields) 
    if "payload" not in field and ("frame" in field or "eth" in field or "ip" in field or "tcp" in field)]
tcp_biased_avoid_field_indexes = [i for i, field in enumerate(fields) 
    if "payload" not in field and "port" not in field and ("frame" in field or "ip" in field or "tcp" in field)]
udp_field_indexes = [i for i, field in enumerate(fields) 
    if "payload" not in field and ("frame" in field or "eth" in field or "ip" in field or "udp" in field or "data" in field)]
udp_biased_avoid_field_indexes = [i for i, field in enumerate(fields) 
    if "payload" not in field and "port" not in field and ("frame" in field or "ip" in field or "udp" in field)]
# print([fields[i] for i in tcp_field_indexes])
# print([fields[i] for i in tcp_biased_avoid_field_indexes])
# print([fields[i] for i in udp_field_indexes])
# print([fields[i] for i in udp_biased_avoid_field_indexes])

def _bigram_generation(packet_datagram, packet_len=64):
    def cut(obj, sec):
        sec = sec % 4 + sec
        return [obj[i: i + sec] for i in range(0, len(obj), sec)]

    result = ""
    generated_datagram = cut(packet_datagram, 1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = (
                    generated_datagram[sub_string_index]
                    + generated_datagram[sub_string_index + 1]
                )
        else:
            break
        result += merge_word_bigram
        result += " "

    return result

def _str_to_ids(text: str, seq_length: int = None, type: str = "bert", CLS_front: bool = False, padding: bool = False):
    if type == "bert":
        tokenizer, SEQ_LENGTH, CLS_TOKEN, PAD_TOKEN, CLS_ID, PAD_ID = _bert_tokenizer()
        if seq_length is None:
            seq_length = SEQ_LENGTH
    elif type == "qwen3vl":
        tokenizer, SEQ_LENGTH, CLS_TOKEN, PAD_TOKEN, CLS_ID, PAD_ID = _Qwen3VL_tokenizer()
        if seq_length is None:
            seq_length = SEQ_LENGTH
    tokens = tokenizer.tokenize(text)
    if CLS_front:
        tokens = [CLS_TOKEN] + tokens
    token_len = len(tokens)
    if padding:
        assert len(tokens) <= seq_length
        tokens = tokens + [PAD_TOKEN] * (seq_length - len(tokens))
    return tokenizer.convert_tokens_to_ids(tokens), token_len

def _ids_to_str(ids, type: str = "bert"):
    if type == "bert":
        tokenizer, SEQ_LENGTH, CLS_TOKEN, PAD_TOKEN, CLS_ID, PAD_ID = _bert_tokenizer()
    elif type == "qwen3vl":
        tokenizer, SEQ_LENGTH, CLS_TOKEN, PAD_TOKEN, CLS_ID, PAD_ID = _Qwen3VL_tokenizer()
    return tokenizer.decode(ids, skip_special_tokens=False)

def _pad(token_ids_list, seqlen, type: str = "bert"):
    if type == "bert":
        tokenizer, SEQ_LENGTH, CLS_TOKEN, PAD_TOKEN, CLS_ID, PAD_ID = _bert_tokenizer()
    elif type == "qwen3vl":
        tokenizer, SEQ_LENGTH, CLS_TOKEN, PAD_TOKEN, CLS_ID, PAD_ID = _Qwen3VL_tokenizer()
    ret = [token_ids + [PAD_ID] * (seqlen - len(token_ids)) for token_ids in token_ids_list]
    # for token_ids in ret:
    #     print(len(token_ids), end="  ")
    # print()
    return ret

def _random_normal_ip(ip_version, ip_that_could_not_be):
    """
    生成一个随机的、非特殊的IP地址。
    
    Args:
        ip_version: IP版本，"4" 或 "6"
        ip_that_could_not_be: 不能生成的IP地址（字符串格式）
        
    Returns:
        随机生成的非特殊IP地址（字符串格式）
    """
    import random
    import ipaddress
    
    # 尝试生成IP的最大次数
    max_attempts = 10
    
    if ip_version == "4":
        # IPv4 需要避免的特殊地址范围
        forbidden_ranges = [
            ipaddress.IPv4Network("127.0.0.0/8"),      # 回环地址
            ipaddress.IPv4Network("169.254.0.0/16"),   # 链路本地地址
            ipaddress.IPv4Network("224.0.0.0/4"),      # 多播地址
            ipaddress.IPv4Network("0.0.0.0/8"),        # 本网络
            ipaddress.IPv4Network("10.0.0.0/8"),       # 私有地址
            ipaddress.IPv4Network("172.16.0.0/12"),    # 私有地址
            ipaddress.IPv4Network("192.168.0.0/16"),   # 私有地址
            ipaddress.IPv4Network("100.64.0.0/10"),    # Carrier-grade NAT
            ipaddress.IPv4Network("198.18.0.0/15"),    # Benchmark testing
            ipaddress.IPv4Network("240.0.0.0/4"),      # 保留地址（E类）
        ]
        
        # 公共DNS服务器和其他特殊IP
        forbidden_ips = [
            "8.8.8.8",           # Google DNS
            "8.8.4.4",           # Google DNS
            "1.1.1.1",           # Cloudflare DNS
            "1.0.0.1",           # Cloudflare DNS
            "255.255.255.255",   # 广播地址
            "0.0.0.0",           # 未指定地址
        ]
        
        # 有效范围：1.0.0.1 到 223.255.255.254（排除所有特殊地址后）
        # 为了简化，我们从公共地址池中随机选择
        # 使用随机生成的地址，然后检查是否在禁止范围内
        
        ip_that_could_not_be_obj = None
        if ip_that_could_not_be is not None:
            try:
                ip_that_could_not_be_obj = ipaddress.IPv4Address(ip_that_could_not_be)
            except ValueError:
                pass
        
        for _ in range(max_attempts):
            # 生成随机IP：在 1.0.0.1 到 223.255.255.254 范围内
            # 但排除已禁止的范围
            # 使用公共地址池：1.0.0.1 - 9.255.255.254, 11.0.0.1 - 126.255.255.254,
            # 128.0.0.1 - 168.253.255.254, 170.0.0.1 - 172.15.255.254, 
            # 172.32.0.1 - 192.167.255.254, 192.169.0.1 - 198.17.255.254,
            # 198.20.0.1 - 223.255.255.254
            parts = [random.randint(1, 223), random.randint(0, 255), 
                    random.randint(0, 255), random.randint(1, 254)]
            ip_str = ".".join(map(str, parts))
            ip = ipaddress.IPv4Address(ip_str)
            
            # 检查是否在禁止范围内
            in_forbidden = any(ip in network for network in forbidden_ranges)
            if in_forbidden:
                continue
            
            # 检查是否是禁止的单个IP
            if ip_str in forbidden_ips:
                continue
            
            # 检查是否等于 ip_that_could_not_be
            if ip_that_could_not_be_obj is not None and ip == ip_that_could_not_be_obj:
                continue
            
            return ip_str
            
        # 如果尝试多次都失败，返回一个默认的公共IP
        return "93.184.216.34"  # example.com 的IP
        
    elif ip_version == "6":
        # IPv6 需要避免的特殊地址范围
        forbidden_ranges = [
            ipaddress.IPv6Network("::1/128"),              # 回环地址
            ipaddress.IPv6Network("fe80::/10"),            # 链路本地地址
            ipaddress.IPv6Network("ff00::/8"),             # 多播地址
            ipaddress.IPv6Network("::/128"),               # 未指定地址
            ipaddress.IPv6Network("::ffff:0:0/96"),        # IPv4映射地址
            ipaddress.IPv6Network("2001:db8::/32"),        # 文档地址
            ipaddress.IPv6Network("2001:10::/28"),         # ORCHID
            ipaddress.IPv6Network("fc00::/7"),             # 唯一本地地址
        ]
        
        # 公共DNS服务器
        forbidden_ips = [
            "2001:4860:4860::8888",      # Google DNS
            "2001:4860:4860::8844",      # Google DNS
            "2606:4700:4700::1111",      # Cloudflare DNS
            "2606:4700:4700::1001",      # Cloudflare DNS
        ]
        
        ip_that_could_not_be_obj = None
        if ip_that_could_not_be is not None:
            try:
                ip_that_could_not_be_obj = ipaddress.IPv6Address(ip_that_could_not_be)
            except ValueError:
                pass
        
        for _ in range(max_attempts):
            # 生成随机IPv6地址（全球单播地址范围：2000::/3）
            # 但排除已禁止的范围
            # 生成 2000:: 到 3fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff 范围内的地址
            # 但排除 2001:db8::/32, 2001:10::/28
            # 简化：生成随机8个16位数字段
            parts_hex = []
            # 第一个段：2000-3fff
            first_seg = random.randint(0x2000, 0x3fff)
            parts_hex.append(first_seg)
            
            # 第二个段
            if first_seg == 0x2001:
                # 如果第一个段是2001，第二个段需要排除 db8-dbff (2001:db8::/32) 和 10 (2001:10::/28)
                # 2001:10::/28 意味着第二个段在 10-1f 范围内（但/28前缀意味着第二个段的低4位可以是任意值）
                # 实际上 2001:10::/28 覆盖了 2001:0010:0000:0000:0000:0000:0000:0000 到 2001:001f:ffff:ffff:ffff:ffff:ffff:ffff
                # 所以第二个段需要排除 0x0010-0x001f
                # 以及 2001:db8::/32 意味着第二个段需要排除 0xdb8-0xdbff (但/32前缀意味着后面6个段都是0)
                # 实际上为了安全，我们排除第二个段为 0xdb8-0xdbff 的所有地址
                second_seg = random.choice([
                    random.randint(0, 0x0f),      # 0-0f (排除 10-1f)
                    random.randint(0x20, 0xdb7),  # 20-db7 (排除 db8-dbff)
                    random.randint(0xdc0, 0xffff)  # dc0-ffff
                ])
            else:
                second_seg = random.randint(0, 0xffff)
            parts_hex.append(second_seg)
            
            # 其余6个段
            for i in range(6):
                parts_hex.append(random.randint(0, 0xffff))
            
            # 转换为字符串格式
            parts = [f"{val:x}" for val in parts_hex]
            ip_str = ":".join(parts)
            ip = ipaddress.IPv6Address(ip_str)
            
            # 检查是否在禁止范围内
            in_forbidden = any(ip in network for network in forbidden_ranges)
            if in_forbidden:
                continue
            
            # 检查是否是禁止的单个IP
            if ip_str.lower() in [f.lower() for f in forbidden_ips]:
                continue
            
            # 检查是否等于 ip_that_could_not_be
            if ip_that_could_not_be_obj is not None and ip == ip_that_could_not_be_obj:
                continue
            
            return ip_str
            
        # 如果尝试多次都失败，返回一个默认的公共IPv6
        return "2606:2800:220:1:248:1893:25c8:1946"  # example.com 的IPv6
        
    else:
        raise ValueError(f"Unsupported IP version: {ip_version}")
    

fields_ids = [_str_to_ids(f"<{field}>", None, "qwen3vl", False, False)[0] for field in fields]
payload_ids = _str_to_ids("<payload>", None, "qwen3vl", False, False)[0]
tcp_payload_ids = _str_to_ids("<tcp.payload>", None, "qwen3vl", False, False)[0]
udp_payload_ids = _str_to_ids("<udp.payload>", None, "qwen3vl", False, False)[0]
def _build_table(lines, payloads, flow_type, extract_payloads_from_lines: bool = False, shuffle_columns: bool = False, random_drop_columns: bool = False, biased_avoid:bool = False):
    table_columnwise = []
    payload_ids_ = None
    if extract_payloads_from_lines:
        payloads = []
    if flow_type is None:
        value0 = lines[0].split("\t")
        if value0[tcp_payload_index] != "":
            flow_type = "TCP"
        elif value0[udp_payload_index] != "":
            flow_type = "UDP"
        else:
            raise ValueError(f"Unknown protocol type: {flow_type}")
    if flow_type == "TCP":
        payload_ids_ = tcp_payload_ids
        valid_indexes = tcp_field_indexes if not biased_avoid else tcp_biased_avoid_field_indexes
        payload_index = tcp_payload_index
    elif flow_type == "UDP":
        payload_ids_ = udp_payload_ids
        valid_indexes = udp_field_indexes if not biased_avoid else udp_biased_avoid_field_indexes
        payload_index = udp_payload_index
    else:
        raise ValueError(f"Unknown protocol type: {flow_type}")
    if lines is not None:
        table_columnwise = [[fields_ids[i]] for i in valid_indexes]
        if biased_avoid:
            ip_dict = {}
        for line in lines:
            values = line.split("\t")
            if biased_avoid:
                ip_src = values[src_index]
                ip_dst = values[dst_index]
                ip_version = values[ip_version_index]
                ip_src_mask = ip_dict.get(ip_src, None)
                ip_dst_mask = ip_dict.get(ip_dst, None)
                if ip_src_mask is None:
                    ip_src_mask = _random_normal_ip(ip_version, ip_dst_mask)
                    ip_dict[ip_src] = ip_src_mask
                if ip_dst_mask is None:
                    ip_dst_mask = _random_normal_ip(ip_version, ip_src_mask)
                    ip_dict[ip_dst] = ip_dst_mask
                values[src_index] = ip_src_mask
                values[dst_index] = ip_dst_mask
            if extract_payloads_from_lines:
                payloads.append(values[payload_index])
            idx = 0
            for i in valid_indexes:
                field_ids = _str_to_ids(f"<{values[i]}>", type="qwen3vl")[0]
                table_columnwise[idx].append(field_ids)
                idx += 1
        if shuffle_columns:
            import random
            random.shuffle(table_columnwise)
        if random_drop_columns:
            import random
            import random
            col_total = len(table_columnwise)
            remain_num = random.randint(col_total // 2, col_total)
            table_columnwise = table_columnwise[:remain_num]
    # print(f"payloads is None: {payloads is None}, payloads: {payloads}")
    payload_bert_ids = []
    if payloads is not None:
        import torch
        table_columnwise.append([payload_ids_])
        for payload in payloads:
            if len(payload) == 0:
                table_columnwise[-1].append(_str_to_ids(f"<>", type="qwen3vl")[0])
            else:
                table_columnwise[-1].append(_str_to_ids(f"<<|image_pad|>>", type="qwen3vl")[0])

                payload_ids, valid_len = _str_to_ids(payload, 1500, "bert", True, True)
                attention_mask_payload = [1]*valid_len+[0]*(1500-valid_len)
                global_attention_mask_payload = [0]*1500
                global_attention_mask_payload[0] = 1
                payload_bert_ids.append((payload_ids, attention_mask_payload, global_attention_mask_payload))
            
    return table_columnwise, payload_bert_ids

def _position_ids(prompt_ids, prompt2_ids, table, label_ids):
    import torch
    prompt_position_ids = torch.arange(len(prompt_ids)).unsqueeze(0).expand(3, -1)
    table_start = prompt_position_ids.max().item()+1
    col_start = table_start
    table_position_ids = []
    h = len(table[0])
    for i, column in enumerate(table):
        seq_len_col = 0
        for cell in column:
            seq_len_col = max(seq_len_col, len(cell))
        pos_0 = torch.full((h, seq_len_col), table_start, dtype=torch.long)
        pos_1 = torch.arange(h).view(-1, 1).expand(-1, seq_len_col)+table_start
        pos_2 = torch.arange(seq_len_col).view(1, -1).expand(h, -1)+col_start
        pos = torch.stack([pos_0, pos_1, pos_2], dim=0).view(3, -1)  # [3, h * seq_len_col]
        to_remove_indexes = []
        for j, cell in enumerate(column):
            len_cell = len(cell)
            if len_cell < seq_len_col:
                to_remove_indexes.extend(list(range(j*seq_len_col+len_cell, (j+1)*seq_len_col)))
        # if i == len(table)-1:
        #     print(f"seq_len_col: {seq_len_col}, to_remove_indexes: {to_remove_indexes}")
        if len(to_remove_indexes) > 0:
            to_remove_indexes = torch.tensor(to_remove_indexes, dtype=torch.long, device=pos.device)
            mask = torch.ones(pos.shape[1], dtype=torch.bool, device=pos.device)
            mask[to_remove_indexes] = False
            pos = pos[:, mask]
        table_position_ids.append(pos)
        col_start += seq_len_col
    labels_start = table_position_ids[-1].max().item()+1
    labels_position_ids = torch.arange(len(prompt2_ids)+len(label_ids)).unsqueeze(0).expand(3, -1) + labels_start
    return torch.cat([prompt_position_ids, torch.cat(table_position_ids, dim=1), labels_position_ids], dim=1)

def _flat_table(table):
    return [item for column in table for cell in column for item in cell]

def _LM_input(lines, payloads, flow_type, label_ids, prompt_ids, prompt2_ids, label = None, extract_payloads_from_lines=False, shuffle_columns=False, random_drop_columns=False, biased_avoid=False,
    _build_table_result=None):
    if _build_table_result is None:
        table, payload_bert_ids = _build_table(lines, payloads, flow_type, extract_payloads_from_lines, shuffle_columns, random_drop_columns, biased_avoid)
    else:
        table, payload_bert_ids = _build_table_result
    position_ids = _position_ids(prompt_ids, prompt2_ids, table, label_ids)
    table_ids = _flat_table(table)
    assert len(prompt_ids)+len(table_ids)+len(prompt2_ids)+len(label_ids) == position_ids.shape[1]
    sample = {
        "data": (prompt_ids+table_ids+prompt2_ids, label_ids, payload_bert_ids, position_ids),
        "label": label
    }
    return sample

def _lines_simplify(lines: list[str]):
    pass

def generate_id2label(tmp_path: str, dest_path: str):
    import os
    import json
    with open(os.path.join(tmp_path, "label_name_to_id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {str(v): k for k, v in label2id.items()}
    with open(os.path.join(dest_path, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

def _cut_stats_list(a: list[tuple]):
    a = sorted(a, key=lambda x: x[0])
    bins = []
    for num, cnt in a[:10]:
        bins.append((num, num, cnt))

    g10_cnt = sum(cnt for _, cnt in a[10:])
    bin_size = g10_cnt // 10
    tmp_cnt = None
    tmp_left = None
    for num, cnt in a[10:]:
        if tmp_cnt is None:
            tmp_left = num
            tmp_cnt = cnt
        else:
            tmp_cnt += cnt
        if tmp_cnt >= bin_size:
            bins.append((tmp_left, num, tmp_cnt))
            tmp_cnt = None
            tmp_left = None
    if tmp_cnt is not None:
        bins.append((tmp_left, a[-1][0], tmp_cnt))
    return bins

def check_tmp(tmp_path: str):
    """
    检查 generate_classify_tmp 生成的sqlite中间文件，输出统计信息。
    Args:
        tmp_path: generate_classify_tmp 生成的临时文件路径
    """
    import os
    import sqlite3

    if not os.path.exists(tmp_path):
        print(f"❌ 目录不存在: {tmp_path}")
        return

    print("=" * 70)
    print("📊 generate_classify_tmp sqlite中间文件统计信息")
    print("=" * 70)

    from .model import connect_to_dbs, close_dbs, execute_sql_on_dbs
    conns = connect_to_dbs(tmp_path)

    # 统计主要表的数量
    flow_count = sum(execute_sql_on_dbs(conns, "SELECT COUNT(*) FROM flows", unpack=True))
    burst_count = sum(execute_sql_on_dbs(conns, "SELECT COUNT(*) FROM bursts", unpack=True))
    packet_count = sum(execute_sql_on_dbs(conns, "SELECT COUNT(*) FROM packets", unpack=True))
    payload_count = sum(execute_sql_on_dbs(conns, "SELECT COUNT(*) FROM payloads", unpack=True))
    label_count = len(execute_sql_on_dbs(conns, "SELECT id FROM labels", unpack=True))

    print(f"\n1️⃣ flows 表: {flow_count}")
    print(f"2️⃣ bursts 表: {burst_count}")
    print(f"3️⃣ packets 表: {packet_count}")
    print(f"4️⃣ payloads 表: {payload_count}")
    print(f"5️⃣ labels 表: {label_count}")

    # flows表 label 分布
    print(f"\n   • flows标签分布（前10）:")
    flows = execute_sql_on_dbs(conns, "SELECT label, COUNT(*) FROM flows GROUP BY label ORDER BY COUNT(*)")
    # INSERT_YOUR_CODE
    # label相同的合并，并取前10
    merged_flows = {}
    for lbl, cnt in flows:
        if lbl in merged_flows:
            merged_flows[lbl] += cnt
        else:
            merged_flows[lbl] = cnt
    # 根据数量降序排序并取前10
    flows = sorted(merged_flows.items(), key=lambda x: x[1], reverse=True)[:10]
    for lbl, cnt in flows:
        print(f"      - {lbl}: {cnt}")
    # flows的payload_ge1_burst_num分布
    print(f"\n   • flows的payload_ge1_burst_num分布（前10）:")
    try:
        flows = execute_sql_on_dbs(conns, "SELECT payload_ge1_burst_num, COUNT(*) FROM flows GROUP BY payload_ge1_burst_num ORDER BY COUNT(*)")
        # INSERT_YOUR_CODE
        # payload_ge1_burst_num相同的合并，并取前10
        merged_flows = {}
        for num, cnt in flows:
            if num in merged_flows:
                merged_flows[num] += cnt
            else:
                merged_flows[num] = cnt
        # 根据数量降序排序并取前10
        # flows = sorted(merged_flows.items(), key=lambda x: x[1], reverse=True)[:10]
        flows = _cut_stats_list(merged_flows.items())
        for min_num, max_num, cnt_sum in flows:
            print(f"      - payload_ge1_burst_num={min_num} ~ {max_num}: {cnt_sum}")
    except Exception as e:
        print(f"      - 获取flows的payload_ge1_burst_num分布异常: {e}")

    # INSERT_YOUR_CODE
    print(f"\n   • flows的packet数量分布（前10）:")
    try:
        # 获取每个flow的packet数量
        flow_packet_counts = execute_sql_on_dbs(
            conns,
            """SELECT SUM(packet_count) AS pkt_num
                FROM bursts
                JOIN burst_packet_count ON bursts.id = burst_packet_count.burst_id
                GROUP BY bursts.flow_id
            """)
        # 统计各packet数量的分布
        packet_num_distribution = {}
        for pkt_num in flow_packet_counts:
            if pkt_num in packet_num_distribution:
                packet_num_distribution[pkt_num] += 1
            else:
                packet_num_distribution[pkt_num] = 1
        # 取前10高频分布
        packet_num_distribution_sorted = _cut_stats_list(packet_num_distribution.items())
        for min_num, max_num, cnt_sum in packet_num_distribution_sorted:
            print(f"      - packet_num={min_num} ~ {max_num}: {cnt_sum}")
    except Exception as e:
        print(f"      - 获取flows的packet数量分布异常: {e}")

    # bursts 表含payload的数量
    print(f"\n   • 含有payload的burst数量（payload_num>0）:")
    burst_payload_num = sum(execute_sql_on_dbs(conns, "SELECT COUNT(*) FROM bursts WHERE payload_num > 0", unpack=True))
    if burst_count > 0:
        percent = burst_payload_num / burst_count * 100
        print(f"      - {burst_payload_num} ({percent:.1f}%)")
    else:
        print(f"      - 0 (0%)")
    # 含payload>=2的burst数量
    print(f"\n   • 含有payload数量≥2的burst数量 (payload_num>=2):")
    burst_payload_ge2_num = sum(execute_sql_on_dbs(conns, "SELECT COUNT(*) FROM bursts WHERE payload_num >= 2", unpack=True))
    if burst_count > 0:
        percent_ge2 = burst_payload_ge2_num / burst_count * 100
        print(f"      - {burst_payload_ge2_num} ({percent_ge2:.1f}%)")
    else:
        print(f"      - 0 (0%)")

    # bursts表 label 分布
    print(f"\n   • bursts标签分布（前10）:")
    bursts = execute_sql_on_dbs(conns, "SELECT label, COUNT(*) FROM bursts GROUP BY label ORDER BY COUNT(*)")
    # INSERT_YOUR_CODE
    # label相同的合并，并取前10
    merged_bursts = {}
    for lbl, cnt in bursts:
        if lbl in merged_bursts:
            merged_bursts[lbl] += cnt
        else:
            merged_bursts[lbl] = cnt
    # 根据数量降序排序并取前10
    bursts = sorted(merged_bursts.items(), key=lambda x: x[1], reverse=True)[:10]
    for lbl, cnt in bursts:
        print(f"      - {lbl}: {cnt}")

    # INSERT_YOUR_CODE
    print(f"\n   • burst payload_num分布（前10）:")
    try:
        burst_payloads = execute_sql_on_dbs(conns, "SELECT payload_num, COUNT(*) FROM bursts GROUP BY payload_num ORDER BY COUNT(*)")
        # payload_num相同的合并，并取前10
        merged_burst_payloads = {}
        for num, cnt in burst_payloads:
            if num in merged_burst_payloads:
                merged_burst_payloads[num] += cnt
            else:
                merged_burst_payloads[num] = cnt
        # 根据数量降序排序并取前10
        burst_payloads_sorted = _cut_stats_list(merged_burst_payloads.items())
        for min_num, max_num, cnt_sum in burst_payloads_sorted:
            print(f"      - payload_num={min_num} ~ {max_num}: {cnt_sum}")
    except Exception as e:
        print(f"      - 获取bursts的payload_num分布异常: {e}")

    # INSERT_YOUR_CODE
    # 统计payload数量>=3、packet数量<=10的bursts数量
    try:
        burst_payload_ge3_packet_le10_count = sum(execute_sql_on_dbs(
            conns,
            """
            SELECT COUNT(*) 
            FROM bursts JOIN burst_packet_count ON bursts.id = burst_packet_count.burst_id
            WHERE bursts.payload_num >= 3 AND burst_packet_count.packet_count <= 10
            """,
            unpack=True
        ))
        print(f"\n   • 满足 payload_num>=3 且 packet_count<=10 的burst数量: {burst_payload_ge3_packet_le10_count}")
    except Exception as e:
        print(f"      - 获取 payload_num>=3 且 packet_count<=10 的burst数量异常: {e}")

    # payloads表长度统计
    print(f"\n   • payload长度分布:")
    try:
        lens = execute_sql_on_dbs(conns, "SELECT LENGTH(content) FROM payloads", unpack=True)
        lens_dict = {}
        for len_ in lens:
            if len_ in lens_dict:
                lens_dict[len_] += 1
            else:
                lens_dict[len_] = 1
        lens_sorted = _cut_stats_list(lens_dict.items())
        for min_len, max_len, cnt_sum in lens_sorted:
            print(f"      - len={min_len} ~ {max_len}: {cnt_sum}")
    except Exception as e:
        print(f"      - 获取payload长度统计异常: {e}")

    # 每个 label 的统计信息
    print(f"\n6️⃣ 各标签统计（label名 | flows | packets | 含payload packets | bursts）:")
    # INSERT_YOUR_CODE
    import os
    import json
    # 假设已定义tmp_path变量，指向tmp目录
    label_name_to_id_path = os.path.join(tmp_path, "label_name_to_id.json")
    with open(label_name_to_id_path, "r", encoding="utf-8") as f:
        label_name_to_id = json.load(f)
    # 构建labels: list of (label_id, label_name)，按id升序排列
    labels = [(int(label_id), label_name) for label_name, label_id in label_name_to_id.items()]
    for label_id, label_name in labels:
        # flows
        n_flows = sum(execute_sql_on_dbs(conns, f"SELECT COUNT(*) FROM label_flows WHERE label_id={label_id}", unpack=True))
        # packets
        n_packets = sum(execute_sql_on_dbs(conns, f"SELECT COUNT(*) FROM label_packets WHERE label_id={label_id}", unpack=True))
        # packets_with_payload
        n_packets_with_payload = sum(execute_sql_on_dbs(conns, f"SELECT COUNT(*) FROM label_packets_with_payload WHERE label_id={label_id}", unpack=True))
        # bursts (这里选label字段 == label_id)
        n_bursts = sum(execute_sql_on_dbs(conns, f"SELECT COUNT(*) FROM bursts WHERE label={label_id}", unpack=True))
        print(f"   - {label_name:20s} | flows: {n_flows:5d} | packets: {n_packets:6d} | packets(pwld): {n_packets_with_payload:6d} | bursts: {n_bursts:5d}")

    close_dbs(conns)

    print("\n" + "=" * 70)
    print("✅ 检查完成！（已采用 sqlite generate_classify_tmp 统计方案）")
    print("=" * 70)
def check_dataset(path: str):
    """
    合并指定目录下所有pickle文件中的数组，并输出合并后数组的长度。
    
    Args:
        directory: 包含pickle文件的目录路径
        
    Returns:
        tuple: (合并后的数组, 数组长度)
    """
    import os
    import pickle
    from tqdm import tqdm
    import gc
    
    if not os.path.exists(path):
        print(f"目录不存在: {path}")
        return None, 0
    
    # 获取所有pickle文件
    pickle_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
    
    if not pickle_files:
        print(f"目录 {path} 中没有找到pickle文件")
        return None, 0
    
    print(f"找到 {len(pickle_files)} 个pickle文件")
    
    merged_array = []
    
    len_dict = {}
    total_len = 0
    for filename in tqdm(pickle_files, desc="合并pickle文件"):
        file_path = os.path.join(path, filename)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    total_len += len(data)
                    len_dict[filename] = total_len
                else:
                    print(f"警告: {filename} 中的数据不是列表类型，跳过")
            del data
            gc.collect()
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")
            continue
    
    print(f"合并完成，总长度: {total_len}")
    # 将len_dict保存到path/len_dict
    import json
    len_dict_path = os.path.join(path, "len_dict.json")
    try:
        with open(len_dict_path, "w", encoding="utf-8") as f:
            json.dump(len_dict, f, ensure_ascii=False, indent=2)
        print(f"已将len_dict写入到 {len_dict_path}")
    except Exception as e:
        print(f"写入len_dict时出错: {e}")

def check_preprocess(preprocess_path: str):
    """
    统计给定路径下各label（子目录）及其各自的样本数量（文件数量）。
    """
    import os

    if not os.path.exists(preprocess_path):
        print(f"指定路径不存在: {preprocess_path}")
        return

    label_names = [name for name in os.listdir(preprocess_path)
                   if os.path.isdir(os.path.join(preprocess_path, name))]
    if not label_names:
        print(f"路径 {preprocess_path} 下未找到任何label子目录")
        return

    print("labels（子目录名称）及其样本数量如下：")
    print("=" * 60)
    total_samples = 0
    for label in sorted(label_names):
        label_dir = os.path.join(preprocess_path, label)
        file_list = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
        num_samples = len(file_list)
        print(f"{label:<30} : {num_samples}")
        total_samples += num_samples
    print("=" * 60)
    print(f"共 {len(label_names)} 个label，总样本数: {total_samples}")

def check_LM_samples(src_path: str):
    """
    检查LM样本内容：如果src_path是目录，则遍历其中所有.pkl文件；如果src_path是文件，则只读取该文件。
    """
    import os
    import pickle
    IMAGE_PAD_ID = 151655

    if not os.path.exists(src_path):
        print(f"❌ 路径不存在: {src_path}")
        return

    from transformers import AutoTokenizer
    model_name_or_path = "./Qwen3-VL-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if os.path.isfile(src_path):
        files = [src_path]
    else:
        files = [os.path.join(src_path, f) for f in os.listdir(src_path) if f.endswith(".pkl")]

    for fpath in sorted(files[:1]):
        try:
            with open(fpath, "rb") as f:
                samples = pickle.load(f)
            for sample in samples[:50]:
                x_ids, y_ids, payloads, position_ids = sample["data"]
                print(len(x_ids), len(y_ids), len(payloads), position_ids.shape)
                assert sum(1 if id_ == IMAGE_PAD_ID else 0 for id_ in x_ids) == len(payloads)
                assert len(x_ids)+len(y_ids) == position_ids.shape[1]
                input_ids =  x_ids+y_ids
                text = tokenizer.decode(input_ids, skip_special_tokens=False)
                print(text)
                position_ids = position_ids.tolist()
                for i in range(3):
                    for j in range(len(position_ids[i])):
                        print(f"{position_ids[i][j]:03d}", end=" ")
                    print()
                print("\n")
        except Exception as e:
            print(f"{fpath}: 读取失败: {e}")

if __name__ == "__main__":
    from fire import Fire
    Fire()