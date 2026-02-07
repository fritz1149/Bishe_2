import os
import sys
from .utils import _cut_bursts, _bigram_generation, _str_to_ids, _dump_in_chunks, fields

def _process_pcap(pcap_path: str, tmp_path: str):

    extract_str = " -e " + " -e ".join(fields) + " "
    cmd = "tshark -r " + pcap_path + extract_str + "-T fields -Y 'tcp or udp' > " + tmp_path
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"tshark command failed with return code {ret}: {cmd}")

def process_flow_dataset(src_path: str, dest_path: str, threads: int = 1):
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
    import concurrent.futures

    def process_single_pcap(args):
        file_path, output_path = args
        _process_pcap(file_path, output_path)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max(1, threads)) as executor:
        futures = [executor.submit(process_single_pcap, args) for args in pcap_files]
        for future in tqdm(as_completed(futures), total=len(pcap_files), desc="处理PCAP文件", unit="file", file=sys.stdout, mininterval=0.1, dynamic_ncols=True):
            future.result()

# def generate_classify1_dataset(tmp_path: str, dest_path: str, k: int = 10000):
#     """
#     生成有标签的数据集：
#     1) 指定 k 作为每类数据的样本数量；
#     2) 从"至少有两个 payload 的 burst"中随机挑选 k 个，每个 burst 取两个带 payload 的行：
#        - 顺序组合为一个样本，label=1；
#        - 逆序组合为一个样本，label=2；
#     3) 从"至少有两个类型2 burst（payload>=1）"的流集合中，随机挑选 k 次：
#        - 每次随机选一个流，再随机选两个 burst；
#        - 各自随机取一个带 payload 的行；
#        - 按 burst 在流中的先后顺序组合为一个样本，label=3；逆序为一个样本，label=4；
#     4) 从"至少有一个类型2 burst"的流中，随机挑选 k 次：
#        - 每次随机选两个不同的流，分别在各自流中随机选一个类型2 burst，随机取一个带 payload 的行；
#        - 两行组成一个样本（顺序任意），label=5；
#     5) 将所有样本使用 _dump_in_chunks 存到 dest_path/classify1 下。
#     """
#     import os
#     import pickle
#     import random
#     from tqdm import tqdm

#     os.makedirs(dest_path, exist_ok=True)
#     out_dir = os.path.join(dest_path, "classify1")
#     os.makedirs(out_dir, exist_ok=True)

#     # 读取临时统计结果
#     print("加载统计结果...")
#     with open(os.path.join(tmp_path, "bursts_payload_ge2.pkl"), "rb") as f:
#         bursts_payload_ge2 = pickle.load(f)
#     with open(os.path.join(tmp_path, "flows_bursts_payload_ge1_count.pkl"), "rb") as f:
#         flows_bursts_payload_ge1_count = pickle.load(f)

#     # 从 flows_bursts_payload_ge1_count 动态得到两个流集合（键为 flow_path，保证可索引）
#     flows_with_ge2 = [
#         flow for flow, detail in flows_bursts_payload_ge1_count.items() if detail["ge1_count"] >= 2
#     ]
#     flows_with_ge1 = [
#         flow for flow, detail in flows_bursts_payload_ge1_count.items() if detail["ge1_count"] >= 1
#     ]

#     samples = []  # 每个样本为 {"lines": [str, str], "label": int}

#     # 类别 1/2：从 bursts_payload_ge2 中采样
#     if bursts_payload_ge2:
#         print("生成类别 1/2 样本...")
#         for _ in tqdm(range(k), desc="类别1/2", leave=False):
#             burst = random.choice(bursts_payload_ge2)
#             payload_lines = burst["payloads"]
#             if len(payload_lines) < 2:
#                 continue
#             i, j = random.sample(range(len(payload_lines)), 2)
#             a = payload_lines[i]
#             b = payload_lines[j]
#             samples.append({"lines": [a, b], "label": 1})
#             samples.append({"lines": [b, a], "label": 2})

#     # 类别 3/4：从拥有 >=2 个类型2 burst 的流中采样
#     if flows_with_ge2:
#         print("生成类别 3/4 样本...")
#         for _ in tqdm(range(k), desc="类别3/4", leave=False):
#             flow = random.choice(flows_with_ge2)
#             bursts = flows_bursts_payload_ge1_count[flow]["bursts_payload_ge1"]
#             if len(bursts) < 2:
#                 continue
#             i, j = random.sample(range(len(bursts)), 2)
#             i, j = (i, j) if i < j else (j, i)
#             b1, b2 = bursts[i], bursts[j]
#             pl1 = b1["payloads"]
#             pl2 = b2["payloads"]
#             if not pl1 or not pl2:
#                 continue
#             l1 = random.choice(pl1)
#             l2 = random.choice(pl2)
#             samples.append({"lines": [l1, l2], "label": 3})
#             samples.append({"lines": [l2, l1], "label": 4})

#     # 类别 5：从拥有 >=1 个类型2 burst 的流中采样，成对不同流
#     if len(flows_with_ge1) >= 2:
#         print("生成类别 5 样本...")
#         for _ in tqdm(range(k), desc="类别5", leave=False):
#             flow_a, flow_b = random.sample(flows_with_ge1, 2)
#             bursts_a = flows_bursts_payload_ge1_count[flow_a]["bursts_payload_ge1"]
#             bursts_b = flows_bursts_payload_ge1_count[flow_b]["bursts_payload_ge1"]
#             if not bursts_a or not bursts_b:
#                 continue
#             ba = random.choice(bursts_a)
#             bb = random.choice(bursts_b)
#             pla = ba["payloads"]
#             plb = bb["payloads"]
#             if not pla or not plb:
#                 continue
#             la = random.choice(pla)
#             lb = random.choice(plb)
#             samples.append({"lines": [la, lb], "label": 5})

#     # 保存所有样本
#     print("保存样本...")
#     _dump_in_chunks(samples, out_dir, k)

    # print(merged_array[:5])
    
    # return merged_array, array_length

if __name__ == '__main__':
    import fire
    fire.Fire()