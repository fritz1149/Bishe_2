import os

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
                "udp.checksum", "udp.checksum.status", "udp.stream", "udp.payload", "data.len"]

def _get_field_index(target):
    for i, field in enumerate(fields):
        if field == target:
            return i
src_index = _get_field_index("ip.src")
tcp_payload_index = _get_field_index("tcp.payload")
udp_payload_index = _get_field_index("udp.payload")


def process_pcap(pcap_path: str, tmp_path: str):

    extract_str = " -e " + " -e ".join(fields) + " "
    cmd = "tshark -r " + pcap_path + extract_str + "-T fields -Y 'tcp or udp' > " + tmp_path
    os.system(cmd)

def process_flow_dataset(src_path: str, dest_path: str):
    """
    处理数据集，将src_path下的pcap文件转换为处理后的文件
    
    Args:
        src_path: 源路径，包含一级文件夹，一级文件夹下包含pcap文件
        dest_path: 目标路径，将创建相同的目录结构并输出处理后的文件
    """
    import os
    import shutil
    from tqdm import tqdm
    
    # 创建目标路径（如果不存在）
    os.makedirs(dest_path, exist_ok=True)
    
    # 收集所有需要处理的pcap文件
    pcap_files = []
    for item in os.listdir(src_path):
        item_path = os.path.join(src_path, item)
        if os.path.isdir(item_path):
            dest_item_path = os.path.join(dest_path, item)
            os.makedirs(dest_item_path, exist_ok=True)
            
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                if os.path.isfile(file_path) and file.lower().endswith('.pcap'):
                    base_name = os.path.splitext(file)[0]
                    output_file = base_name + '.txt'
                    output_path = os.path.join(dest_item_path, output_file)
                    pcap_files.append((file_path, output_path))
    
    # 使用tqdm显示进度
    for file_path, output_path in tqdm(pcap_files, desc="处理PCAP文件"):
        process_pcap(file_path, output_path)

def _cut_bursts(in_path):
    bursts = []
    with open(in_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        src_ip = lines[0][:-1].split("\t")[src_index]
        current_burst = []
        payloads_in_current_burst = []
        payload_num = 0
        for line in lines:
            values = line[:-1].split("\t")
            ip = values[src_index]
            if src_ip != ip:
                bursts.append({"packets": current_burst, "payload_num": payload_num, "payloads": payloads_in_current_burst})
                current_burst = []
                payload_num = 0
                payloads_in_current_burst = []
                src_ip = ip
            current_burst.append(line)
            payload = values[tcp_payload_index]
            if payload == "":
                payload = values[udp_payload_index]
            if payload != "":
                payload_num += 1
                payloads_in_current_burst.append(payload)
        if len(current_burst) > 0:
                bursts.append({"packets": current_burst, "payload_num": payload_num, "payloads": payloads_in_current_burst})
    return bursts

def _dump_in_chunks(items, out_dir, chunk_size):
    import os
    import pickle
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    if chunk_size == -1:
        chunk_size = len(items)
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        file_name = f"part_{idx:05d}.pkl"
        file_path = os.path.join(out_dir, file_name)
        with open(file_path, "wb") as fout:
            pickle.dump(chunk, fout)
        idx += 1

def generate_packet_dataset(src_path: str, dest_path: str, k: int = 10000):
    # 从flow中间数据生成packet数据集
    import os
    import shutil
    from tqdm import tqdm
    
    # 创建目标路径（如果不存在）
    os.makedirs(dest_path, exist_ok=True)
    
    all_packets = []
    # 收集所有txt文件
    txt_files = []
    for item in os.listdir(src_path):
        item_path = os.path.join(src_path, item)
        if os.path.isdir(item_path):
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                if os.path.isfile(file_path) and file.lower().endswith('.txt'):
                    txt_files.append(file_path)
    
    # 使用tqdm显示进度
    for file_path in tqdm(txt_files, desc="处理文本文件"):
        with open(file_path, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            for line in lines:
                # 提取payload内容
                values = line[:-1].split("\t")
                payload = values[tcp_payload_index]
                if payload == "":
                    payload = values[udp_payload_index]
                all_packets.append(payload)
    
    import random
    random.shuffle(all_packets)

    # 按 9:1 划分训练集与测试集
    total_count = len(all_packets)
    train_count = int(total_count * 0.9)
    train_packets = all_packets[:train_count]
    test_packets = all_packets[train_count:]

    # 创建 train/test 目录
    train_dir = os.path.join(dest_path, "train")
    test_dir = os.path.join(dest_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 保存分片pickle
    print("保存训练集...")
    _dump_in_chunks(train_packets, train_dir, k)
    print("保存测试集...")
    _dump_in_chunks(test_packets, test_dir, k)


def generate_flow_dataset(src_path: str, dest_path: str, k: int = 1000):
    """
    生成流级别、带标签的数据集。

    - 源目录结构：src_path/<label>/*.txt（每个 txt 为一个流的中间结果）
    - 目标目录结构：
        dest_path/train/<label>/part_XXXXX.pkl
        dest_path/test/<label>/part_XXXXX.pkl
    - 每个样本为一个"嵌套列表"（即该 txt 文件的所有行构成的列表）
    - 每个 pkl 中存放最多 k 个样本（默认 10000）
    - 每个标签内独立按 9:1 划分训练/测试
    """
    import os
    import pickle
    import random
    from tqdm import tqdm

    # 创建目标根路径
    os.makedirs(dest_path, exist_ok=True)

    # 目标下的 train/test 目录
    train_root = os.path.join(dest_path, "train")
    test_root = os.path.join(dest_path, "test")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    # 获取所有标签
    labels = [item for item in os.listdir(src_path) 
              if os.path.isdir(os.path.join(src_path, item))]

    # 遍历标签（src_path 的一级目录）
    for label in tqdm(labels, desc="处理标签"):
        label_src_dir = os.path.join(src_path, label)
        if not os.path.isdir(label_src_dir):
            continue

        # 收集该标签下的所有样本（每个样本为一个 txt 的行列表）
        samples = []
        txt_files = [f for f in os.listdir(label_src_dir) 
                     if f.lower().endswith('.txt')]
        
        for fname in tqdm(txt_files, desc=f"处理标签 {label}", leave=False):
            fpath = os.path.join(label_src_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r", encoding="utf-8") as fin:
                    lines = fin.readlines()
                    samples.append(lines)

        if not samples:
            continue

        # 每个标签内随机打乱并 9:1 划分
        random.shuffle(samples)
        total = len(samples)
        train_n = int(total * 0.9)
        train_samples = samples[:train_n]
        test_samples = samples[train_n:]

        # 该标签在 train/test 下的目录
        train_label_dir = os.path.join(train_root, label)
        test_label_dir = os.path.join(test_root, label)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

        # 写入分片 pickle（嵌套列表：每个元素为一个 txt 的行列表）
        print(f"保存标签 {label} 的训练集...")
        _dump_in_chunks(train_samples, train_label_dir, k)
        print(f"保存标签 {label} 的测试集...")
        _dump_in_chunks(test_samples, test_label_dir, k)


def generate_classify_tmp(src_path: str, tmp_path: str):
    """
    对所有流（不分标签）进行统计并在内存中保存相关数据结构：
    1) 统计所有"至少包含两个 payload 的 burst"（全局），并保存这些 burst；
    2) 统计每个流中"至少包含一个 payload 的 burst"的数量；
    3) 统计"至少有两个类型2（payload>=1）burst"的流集合；
    4) 统计"至少有一个类型2（payload>=1）burst"的流集合。
    5) 新增：统计所有payload内容，格式为{label: [payload, ...]}，并保存

    返回包含上述统计与数据结构的字典。
    """
    import os
    from tqdm import tqdm

    bursts_payload_ge2 = []  # 全局：满足 payload_num>=2 的 burst
    flows_bursts_payload_ge1_count = {}  # flow_path -> 满足 payload_num>=1 的 burst 数量

    # 新增：统计，每个label对应一个payload列表
    payload_label_dict = {}

    # 收集所有流文件
    flow_files = []
    for label in os.listdir(src_path):
        label_dir = os.path.join(src_path, label)
        if os.path.isdir(label_dir):
            for fname in os.listdir(label_dir):
                if fname.lower().endswith('.txt'):
                    flow_path = os.path.join(label_dir, fname)
                    if os.path.isfile(flow_path):
                        flow_files.append((flow_path, label))

    # 使用tqdm显示进度
    for flow_path, label in tqdm(flow_files, desc="分析流文件"):
        if label not in payload_label_dict:
            payload_label_dict[label] = []
        bursts = _cut_bursts(flow_path)
        ge1_count = 0
        bursts_payload_ge1 = []
        for b in bursts:
            # 新增：把payload记录到payload_label_dict中
            if "payloads" in b and isinstance(b["payloads"], list):
                for payload in b["payloads"]:
                    if payload.strip():  # 只统计非空payload
                        payload_label_dict[label].append(payload)

            # 统计burst数量
            if b["payload_num"] >= 2:
                bursts_payload_ge2.append(b)
            if b["payload_num"] >= 1:
                ge1_count += 1
                bursts_payload_ge1.append(b)
        flows_bursts_payload_ge1_count[flow_path] = {"ge1_count": ge1_count, "bursts_payload_ge1": bursts_payload_ge1}

    # 将结果保存到 tmp_path
    import pickle
    os.makedirs(tmp_path, exist_ok=True)

    print("保存统计结果...")
    with open(os.path.join(tmp_path, "bursts_payload_ge2.pkl"), "wb") as f:
        pickle.dump(bursts_payload_ge2, f)
    with open(os.path.join(tmp_path, "flows_bursts_payload_ge1_count.pkl"), "wb") as f:
        pickle.dump(flows_bursts_payload_ge1_count, f)
    # 保存payload与label的统计字典
    with open(os.path.join(tmp_path, "payload_label_dict.pkl"), "wb") as f:
        pickle.dump(payload_label_dict, f)
def generate_classify1_dataset(tmp_path: str, dest_path: str, k: int = 10000):
    """
    生成有标签的数据集：
    1) 指定 k 作为每类数据的样本数量；
    2) 从"至少有两个 payload 的 burst"中随机挑选 k 个，每个 burst 取两个带 payload 的行：
       - 顺序组合为一个样本，label=1；
       - 逆序组合为一个样本，label=2；
    3) 从"至少有两个类型2 burst（payload>=1）"的流集合中，随机挑选 k 次：
       - 每次随机选一个流，再随机选两个 burst；
       - 各自随机取一个带 payload 的行；
       - 按 burst 在流中的先后顺序组合为一个样本，label=3；逆序为一个样本，label=4；
    4) 从"至少有一个类型2 burst"的流中，随机挑选 k 次：
       - 每次随机选两个不同的流，分别在各自流中随机选一个类型2 burst，随机取一个带 payload 的行；
       - 两行组成一个样本（顺序任意），label=5；
    5) 将所有样本使用 _dump_in_chunks 存到 dest_path/classify1 下。
    """
    import os
    import pickle
    import random
    from tqdm import tqdm

    os.makedirs(dest_path, exist_ok=True)
    out_dir = os.path.join(dest_path, "classify1")
    os.makedirs(out_dir, exist_ok=True)

    # 读取临时统计结果
    print("加载统计结果...")
    with open(os.path.join(tmp_path, "bursts_payload_ge2.pkl"), "rb") as f:
        bursts_payload_ge2 = pickle.load(f)
    with open(os.path.join(tmp_path, "flows_bursts_payload_ge1_count.pkl"), "rb") as f:
        flows_bursts_payload_ge1_count = pickle.load(f)

    # 从 flows_bursts_payload_ge1_count 动态得到两个流集合（键为 flow_path，保证可索引）
    flows_with_ge2 = [
        flow for flow, detail in flows_bursts_payload_ge1_count.items() if detail["ge1_count"] >= 2
    ]
    flows_with_ge1 = [
        flow for flow, detail in flows_bursts_payload_ge1_count.items() if detail["ge1_count"] >= 1
    ]

    samples = []  # 每个样本为 {"lines": [str, str], "label": int}

    # 类别 1/2：从 bursts_payload_ge2 中采样
    if bursts_payload_ge2:
        print("生成类别 1/2 样本...")
        for _ in tqdm(range(k), desc="类别1/2", leave=False):
            burst = random.choice(bursts_payload_ge2)
            payload_lines = burst["payloads"]
            if len(payload_lines) < 2:
                continue
            i, j = random.sample(range(len(payload_lines)), 2)
            a = payload_lines[i]
            b = payload_lines[j]
            samples.append({"lines": [a, b], "label": 1})
            samples.append({"lines": [b, a], "label": 2})

    # 类别 3/4：从拥有 >=2 个类型2 burst 的流中采样
    if flows_with_ge2:
        print("生成类别 3/4 样本...")
        for _ in tqdm(range(k), desc="类别3/4", leave=False):
            flow = random.choice(flows_with_ge2)
            bursts = flows_bursts_payload_ge1_count[flow]["bursts_payload_ge1"]
            if len(bursts) < 2:
                continue
            i, j = random.sample(range(len(bursts)), 2)
            i, j = (i, j) if i < j else (j, i)
            b1, b2 = bursts[i], bursts[j]
            pl1 = b1["payloads"]
            pl2 = b2["payloads"]
            if not pl1 or not pl2:
                continue
            l1 = random.choice(pl1)
            l2 = random.choice(pl2)
            samples.append({"lines": [l1, l2], "label": 3})
            samples.append({"lines": [l2, l1], "label": 4})

    # 类别 5：从拥有 >=1 个类型2 burst 的流中采样，成对不同流
    if len(flows_with_ge1) >= 2:
        print("生成类别 5 样本...")
        for _ in tqdm(range(k), desc="类别5", leave=False):
            flow_a, flow_b = random.sample(flows_with_ge1, 2)
            bursts_a = flows_bursts_payload_ge1_count[flow_a]["bursts_payload_ge1"]
            bursts_b = flows_bursts_payload_ge1_count[flow_b]["bursts_payload_ge1"]
            if not bursts_a or not bursts_b:
                continue
            ba = random.choice(bursts_a)
            bb = random.choice(bursts_b)
            pla = ba["payloads"]
            plb = bb["payloads"]
            if not pla or not plb:
                continue
            la = random.choice(pla)
            lb = random.choice(plb)
            samples.append({"lines": [la, lb], "label": 5})

    # 保存所有样本
    print("保存样本...")
    _dump_in_chunks(samples, out_dir, k)

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

def _str_to_ids(text: str, seq_length: int, tokenizer):
    from uer.uer.utils import CLS_TOKEN, PAD_TOKEN
    tokens = tokenizer.tokenize(text)
    token_len = len(tokens)+1
    assert len(tokens) <= seq_length
    tokens = [CLS_TOKEN] + tokens + [PAD_TOKEN] * (seq_length - len(tokens))
    return tokenizer.convert_tokens_to_ids(tokens), token_len

def generate_contrastive_dataset(tmp_path: str, dest_path: str, k: int = 10000, k2: int = 1000):
    """
    生成对比学习数据集（简化版本，只有两类标签），并按9:1划分训练集和测试集。
    额外：加载payload_label_dict，从每个label中随机选取k2个payload，将<payload, label>对打乱并保存在dest_path/test路径下，用_dump_in_chunks分片保存。
    
    Args:
        tmp_path: generate_classify1_tmp 生成的临时文件路径
        dest_path: 输出数据集路径
        k: 每类样本数量（对比训练集）
        k2: 每类测试集payload数量
    """
    import os
    import pickle
    import random
    from tqdm import tqdm

    out_dir = dest_path
    os.makedirs(out_dir, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    from uer.uer.opts import tokenizer_opts
    tokenizer_opts(parser)
    args = parser.parse_args([])  # 使用空列表，采用默认参数而非命令行
    args.vocab_path = "config/encryptd_vocab.txt"
    from uer.uer.utils import str2tokenizer
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.vocab = args.tokenizer.vocab
    SEQ_LENGTH = 1500

    # 读取临时统计结果
    print("加载统计结果...")
    with open(os.path.join(tmp_path, "bursts_payload_ge2.pkl"), "rb") as f:
        bursts_payload_ge2 = pickle.load(f)
    with open(os.path.join(tmp_path, "flows_bursts_payload_ge1_count.pkl"), "rb") as f:
        flows_bursts_payload_ge1_count = pickle.load(f)

    # 从 flows_bursts_payload_ge1_count 动态得到流集合
    flows_with_ge2 = [
        flow for flow, detail in flows_bursts_payload_ge1_count.items() if detail["ge1_count"] >= 2
    ]

    samples_label1 = []  # 类别1的样本
    samples_label2 = []  # 类别2的样本

    # 类别 1：从 bursts_payload_ge2 中采样（同一 burst 内的正样本对）
    if bursts_payload_ge2:
        print("生成类别 1 样本（同 burst 内）...")
        for _ in tqdm(range(k), desc="类别1", leave=False):
            burst = random.choice(bursts_payload_ge2)
            payload_lines = burst["payloads"]
            assert len(payload_lines) >= 2
            i, j = random.sample(range(len(payload_lines)), 2)
            a = _str_to_ids(_bigram_generation(payload_lines[i]), SEQ_LENGTH, args.tokenizer)
            b = _str_to_ids(_bigram_generation(payload_lines[j]), SEQ_LENGTH, args.tokenizer)
            # 随机选择顺序
            if random.random() < 0.5:
                samples_label1.append({"data": [a, b], "label": 1})
            else:
                samples_label1.append({"data": [b, a], "label": 1})

    # 类别 2：从拥有 >=2 个类型2 burst 的流中采样（同流内不同 burst）
    if flows_with_ge2:
        print("生成类别 2 样本（同流内不同 burst）...")
        for _ in tqdm(range(k), desc="类别2", leave=False):
            flow = random.choice(flows_with_ge2)
            bursts = flows_bursts_payload_ge1_count[flow]["bursts_payload_ge1"]
            assert len(bursts) >= 2
            i, j = random.sample(range(len(bursts)), 2)
            b1, b2 = bursts[i], bursts[j]
            pl1 = b1["payloads"]
            pl2 = b2["payloads"]
            assert len(pl1) >= 1 and len(pl2) >= 1
            l1 = _str_to_ids(_bigram_generation(random.choice(pl1)), SEQ_LENGTH, args.tokenizer)
            l2 = _str_to_ids(_bigram_generation(random.choice(pl2)), SEQ_LENGTH, args.tokenizer)
            # 随机选择顺序
            if random.random() < 0.5:
                samples_label2.append({"data": [l1, l2], "label": 2})
            else:
                samples_label2.append({"data": [l2, l1], "label": 2})

    # 对每个类别进行随机打乱
    random.shuffle(samples_label1)
    random.shuffle(samples_label2)

    # 所有样本归为训练集
    train_samples_1 = samples_label1
    train_samples_2 = samples_label2
    train_samples = train_samples_1 + train_samples_2

    # 打乱训练样本
    random.shuffle(train_samples)

    # 创建输出目录（train）
    train_dir = os.path.join(out_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    # 保存训练集
    print(f"保存训练集（{len(train_samples)} 个样本）...")
    _dump_in_chunks(train_samples, train_dir, k)
    print(f"对比学习数据集生成完成！")
    print(f"  类别1: 训练集 {len(train_samples_1)} 个")
    print(f"  类别2: 训练集 {len(train_samples_2)} 个")
    print(f"  总计: 训练集 {len(train_samples)} 个")

    # ----- 新增逻辑: 生成单条payload测试集，并分片保存 -----
    # 加载payload_label_dict
    payload_label_path = os.path.join(tmp_path, "payload_label_dict.pkl")
    if os.path.exists(payload_label_path):
        with open(payload_label_path, "rb") as f:
            payload_label_dict = pickle.load(f)
        print(f"已加载 payload_label_dict（标签数: {len(payload_label_dict)}）")
    else:
        print(f"未找到payload_label_dict.pkl，跳过测试集生成")
        return

    # 将字符串label映射为int，并生成映射表
    test_samples = []
    label2id = {label: idx for idx, label in enumerate(sorted(payload_label_dict.keys()))}
    id2label = {idx: label for label, idx in label2id.items()}
    for label, payloads in payload_label_dict.items():
        if not payloads:
            continue
        if len(payloads) <= k2:
            chosen_payloads = payloads[:]  # 全部
        else:
            chosen_payloads = random.sample(payloads, k2)
        label_int = label2id[label]
        for payload in chosen_payloads:
            test_samples.append({"data": _str_to_ids(_bigram_generation(payload), SEQ_LENGTH, args.tokenizer), "label": label_int})

    # 打乱
    random.shuffle(test_samples)

    # 保存到 dest_path/test，分片
    test_dir = os.path.join(out_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    print(f"保存单条payload测试集（{len(test_samples)} 个样本）到: {test_dir}")
    _dump_in_chunks(test_samples, test_dir, k2)

    # 保存label映射表
    label2id_path = os.path.join(test_dir, "label2id.json")
    id2label_path = os.path.join(test_dir, "id2label.json")
    with open(label2id_path, "w", encoding="utf-8") as f:
        import json
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(id2label_path, "w", encoding="utf-8") as f:
        import json
        json.dump(id2label, f, ensure_ascii=False, indent=2)

def check_tmp(tmp_path: str):
    """
    检查 generate_classify1_tmp 生成的中间文件，输出统计信息。
    
    Args:
        tmp_path: generate_classify1_tmp 生成的临时文件路径
    """
    import os
    import pickle
    
    if not os.path.exists(tmp_path):
        print(f"❌ 目录不存在: {tmp_path}")
        return
    
    print("=" * 70)
    print("📊 中间文件统计信息")
    print("=" * 70)
    
    # 检查 bursts_payload_ge2.pkl
    bursts_file = os.path.join(tmp_path, "bursts_payload_ge2.pkl")
    if os.path.exists(bursts_file):
        with open(bursts_file, "rb") as f:
            bursts_payload_ge2 = pickle.load(f)
        
        print(f"\n1️⃣  bursts_payload_ge2.pkl")
        print(f"   • 总 burst 数量: {len(bursts_payload_ge2)}")
        
        if bursts_payload_ge2:
            # 统计 payload 数量分布
            payload_counts = [b["payload_num"] for b in bursts_payload_ge2]
            print(f"   • Payload 数量范围: {min(payload_counts)} ~ {max(payload_counts)}")
            print(f"   • 平均 Payload 数量: {sum(payload_counts) / len(payload_counts):.2f}")
            
            # 统计不同 payload 数量的 burst 分布
            from collections import Counter
            count_dist = Counter(payload_counts)
            print(f"   • Payload 数量分布（前5）:")
            for count, freq in sorted(count_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      - {count} 个 payload: {freq} 个 burst")
            
            # 示例数据
            sample = bursts_payload_ge2[0]
            print(f"   • 示例 burst 结构:")
            print(f"      - packets 数量: {len(sample['packets'])}")
            print(f"      - payloads 数量: {len(sample['payloads'])}")
    else:
        print(f"\n❌ 文件不存在: bursts_payload_ge2.pkl")
    
    # 检查 flows_bursts_payload_ge1_count.pkl
    flows_file = os.path.join(tmp_path, "flows_bursts_payload_ge1_count.pkl")
    if os.path.exists(flows_file):
        with open(flows_file, "rb") as f:
            flows_bursts_payload_ge1_count = pickle.load(f)
        
        print(f"\n2️⃣  flows_bursts_payload_ge1_count.pkl")
        print(f"   • 总流数量: {len(flows_bursts_payload_ge1_count)}")
        
        if flows_bursts_payload_ge1_count:
            # 统计每个流的 burst 数量
            ge1_counts = [detail["ge1_count"] for detail in flows_bursts_payload_ge1_count.values()]
            print(f"   • 类型2 burst 数量范围: {min(ge1_counts)} ~ {max(ge1_counts)}")
            print(f"   • 平均类型2 burst 数量: {sum(ge1_counts) / len(ge1_counts):.2f}")
            
            # 统计满足条件的流
            flows_with_ge2 = sum(1 for count in ge1_counts if count >= 2)
            flows_with_ge1 = sum(1 for count in ge1_counts if count >= 1)
            
            print(f"   • 至少有 1 个类型2 burst 的流: {flows_with_ge1}")
            print(f"   • 至少有 2 个类型2 burst 的流: {flows_with_ge2}")
            
            # 类型2 burst 数量分布
            from collections import Counter
            count_dist = Counter(ge1_counts)
            print(f"   • 类型2 burst 数量分布（前5）:")
            for count, freq in sorted(count_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      - {count} 个类型2 burst: {freq} 个流")
            
            # 示例数据
            sample_flow = list(flows_bursts_payload_ge1_count.keys())[0]
            sample_detail = flows_bursts_payload_ge1_count[sample_flow]
            print(f"   • 示例流结构:")
            print(f"      - ge1_count: {sample_detail['ge1_count']}")
            print(f"      - bursts_payload_ge1 数量: {len(sample_detail['bursts_payload_ge1'])}")
    else:
        print(f"\n❌ 文件不存在: flows_bursts_payload_ge1_count.pkl")
    
    # INSERT_YOUR_CODE
    # 检查 payload_dict.pkl
    payload_dict_file = os.path.join(tmp_path, "payload_label_dict.pkl")
    if os.path.exists(payload_dict_file):
        with open(payload_dict_file, "rb") as f:
            payload_dict = pickle.load(f)

        print(f"\n3️⃣  payload_label_dict.pkl")
        print(f"   • 键值对数量: {len(payload_dict)}")

        if payload_dict:
            # 获取所有payload长度
            payload_lengths = []
            for v in payload_dict.values():
                payload_lengths.extend(len(p) for p in v)

            if payload_lengths:
                print(f"   • payload长度范围: {min(payload_lengths)} ~ {max(payload_lengths)}")
                from collections import Counter
                payload_len_dist = Counter(payload_lengths)
                print(f"   • payload长度分布（前5）:")
                for l, freq in payload_len_dist.most_common(5):
                    print(f"      - 长度 {l}: {freq} 个")
            else:
                print(f"   • 没有可统计长度的payload")
    else:
        print(f"\n❌ 文件不存在: payload_label_dict.pkl")
    
    print("\n" + "=" * 70)
    print("✅ 检查完成！")
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
    
    # 使用tqdm显示进度
    for filename in tqdm(pickle_files, desc="合并pickle文件"):
        file_path = os.path.join(path, filename)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    merged_array.extend(data)
                else:
                    print(f"警告: {filename} 中的数据不是列表类型，跳过")
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")
            continue
    
    array_length = len(merged_array)
    print(f"合并完成，总长度: {array_length}")
    # print(merged_array[:5])
    
    # return merged_array, array_length


def test(arg: str):
    print(f'test: {arg}')

def main():
    """
    使用 Fire 库管理命令行参数的主函数。
    
    支持的函数调用：
    - process_pcap(pcap_path, tmp_path)
    - process_flow_dataset(src_path, dest_path)
    - generate_packet_dataset(src_path, dest_path, k=10000)
    - generate_flow_dataset(src_path, dest_path, k=1000)
    - generate_classify1_tmp(src_path, tmp_path)
    - generate_classify1_dataset(tmp_path, dest_path, k=10000)
    - generate_contrastive_dataset(tmp_path, dest_path, k=10000)
    - check_tmp(tmp_path)
    - check_dataset(path)
    
    使用示例：
    python preprocess.py process_pcap --pcap_path="input.pcap" --tmp_path="output.txt"
    python preprocess.py generate_packet_dataset --src_path="flows" --dest_path="packets" --k=5000
    python preprocess.py generate_classify1_tmp --src_path="flows" --tmp_path="tmp"
    python preprocess.py check_tmp --tmp_path="tmp"
    python preprocess.py generate_contrastive_dataset --tmp_path="tmp" --dest_path="output" --k=10000
    python preprocess.py check_dataset --path="path/to/pickle/files"
    """
    import fire
    fire.Fire()


if __name__ == "__main__":
    main()
