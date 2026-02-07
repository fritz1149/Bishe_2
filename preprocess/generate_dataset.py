def generate_contrastive_dataset(tmp_path: str, dest_path: str, k1: int = 10000, k2: int = 1000, k3: int = 1000):
    """
    生成对比学习数据集（简化版本，只有两类标签），分别保存类别1和类别2样本。类别1先处理保存，释放变量，后处理类别2。
    额外：从payloads_list和labels_list构建payload_label_dict，从每个label中随机选取k3个payload，将<payload, label>对打乱并保存在dest_path/test路径下，用_dump_in_chunks分片保存。
    
    使用新版generate_classify_tmp产生的中间文件：
    - bursts_list.pkl: Burst对象列表
    - flows_list.pkl: Flow对象列表  
    - payloads_list.pkl: Payload对象列表
    - labels_list.pkl: Label对象列表
    """
    import os
    import pickle
    import random
    from tqdm import tqdm
    import gc
    from collections import defaultdict
    from .utils import _bigram_generation, _str_to_ids, _dump_in_chunks

    out_dir = dest_path
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, "train")
    
    from .model import connect_to_dbs, execute_sql_on_dbs, close_dbs
    dbs = connect_to_dbs(tmp_path)

    # 新逻辑: 由于id就是下标，直接索引即可，无需dict
    # -------------------------------
    # 1. 从bursts_list中筛选payload_num >= 2的burst，用于类别1正样本生成
    if k1 > 0:
        samples_label1 = []
        print("筛选 payload_num >= 2 的 burst ...")
        selected_bursts = execute_sql_on_dbs(dbs, f"SELECT id FROM bursts WHERE payload_num >= 2 ORDER BY RANDOM() LIMIT {k1//len(dbs)+1}", unpack=True)
        for burst_id in tqdm(selected_bursts, desc="类别1", leave=False):
            payload_ids = execute_sql_on_dbs(dbs, f"SELECT payload_id FROM burst_payloads WHERE burst_id = {burst_id}", unpack=True)
            assert len(payload_ids) >= 2
            selected_payload_ids = random.sample(payload_ids, 2)
            selected_payload_contents = execute_sql_on_dbs(dbs, f"SELECT content FROM payloads WHERE id IN ({','.join([str(id) for id in selected_payload_ids])})", unpack=True)
            a = _str_to_ids(
                text = _bigram_generation(selected_payload_contents[0]),
                seq_length = None,
                type = "bert",
                CLS_front = True,
                padding = True
            )
            b = _str_to_ids(
                text = _bigram_generation(selected_payload_contents[1]),
                seq_length = None,
                type = "bert",
                CLS_front = True,
                padding = True
            )
            samples_label1.append({"data": [a, b], "label": 1})
        # --- 保存类别1数据 ---
        random.shuffle(samples_label1)
        print(f"保存类别1训练集（{len(samples_label1)} 个样本）...")
        _dump_in_chunks(samples_label1, train_dir, k1, name="class1")
        print(f"类别1训练集保存完成！ 样本数: {len(samples_label1)}")
        # 保存完立即释放
        count_class1 = len(samples_label1)
        del samples_label1
        gc.collect()

    # -------------------------------
    # 2. 从flows_list中构建流统计信息，用于类别2正样本生成
    if k2 > 0:
        print("构建流统计信息...")
        flows_with_ge2_le10 = execute_sql_on_dbs(dbs, f"SELECT id FROM flows WHERE payload_ge1_burst_num >= 2 AND payload_ge1_burst_num <= 10", unpack=True)
        flows_with_g10 = execute_sql_on_dbs(dbs, f"SELECT id, payload_ge1_burst_num FROM flows WHERE payload_ge1_burst_num > 10")
        # assert len(flows_with_ge2_le10)+len(flows_with_g10) >= k2, f"可用流数量不足，当前 {len(flows_with_ge2_le10)+len(flows_with_g10)}，需 {k2}"
        samples_label2 = []

        if flows_with_ge2_le10:
            print("生成类别 2 样本（同流内不同 burst）...")
            k2_1 = min(int(k2), int(len(flows_with_ge2_le10)))
            selected_flows = random.sample(flows_with_ge2_le10, k2_1)
            for flow_id in tqdm(selected_flows, desc="类别2", leave=False):
                # 随机选择该flow中的两个burst 
                burst_ids = execute_sql_on_dbs(dbs, f"SELECT id FROM bursts WHERE flow_id = {flow_id} AND payload_num >= 1 ORDER BY RANDOM() LIMIT 2", unpack=True)
                packet_ids = []
                for burst_id in burst_ids:
                    burst_payload_ids = execute_sql_on_dbs(dbs, f"SELECT payload_id FROM burst_payloads WHERE burst_id = {burst_id} ORDER BY RANDOM() LIMIT 1", unpack=True)
                    packet_ids.append(burst_payload_ids[0])
                b = execute_sql_on_dbs(dbs, f"SELECT content FROM payloads WHERE id IN ({','.join([str(id) for id in packet_ids])})", unpack=True)
                l1 = _str_to_ids(
                    text = _bigram_generation(b[0]),
                    seq_length = None,
                    type = "bert",
                    CLS_front = True,
                    padding = True
                )
                l2 = _str_to_ids(
                    text = _bigram_generation(b[1]),
                    seq_length = None,
                    type = "bert",
                    CLS_front = True,
                    padding = True
                )
                samples_label2.append({"data": [l1, l2], "label": 2})   

        if k2 > k2_1 and flows_with_g10:
            k2_2 = k2 - k2_1
            # 让burst数量多的流被选中的概率更大，且权重为 n(n-1)，n为ge1_count
            weights = []
            for flow_id, n in flows_with_g10:
                weights.append(n * (n - 1))
            for _ in tqdm(range(k2_2), desc="类别2", leave=False):
                selected_flow = random.choices(flows_with_g10, weights=weights, k=1)[0][0]
                # print(f"SELECT id FROM bursts WHERE flow_id = {selected_flow} ORDER BY RANDOM() LIMIT 2")
                burst_ids = execute_sql_on_dbs(dbs, f"SELECT id FROM bursts WHERE flow_id = {selected_flow} AND payload_num >= 1 ORDER BY RANDOM() LIMIT 2", unpack=True)
                packet_ids = []
                for burst_id in burst_ids:
                    burst_payload_ids = execute_sql_on_dbs(dbs, f"SELECT payload_id FROM burst_payloads WHERE burst_id = {burst_id} ORDER BY RANDOM() LIMIT 1", unpack=True)
                    packet_ids.append(burst_payload_ids[0])
                b = execute_sql_on_dbs(dbs, f"SELECT content FROM payloads WHERE id IN ({','.join([str(id) for id in packet_ids])})", unpack=True)
                l1 = _str_to_ids(
                    text = _bigram_generation(b[0]),
                    seq_length = None,
                    type = "bert",
                    CLS_front = True,
                    padding = True
                )
                l2 = _str_to_ids(
                    text = _bigram_generation(b[1]),
                    seq_length = None,
                    type = "bert",
                    CLS_front = True,
                    padding = True
                )
                samples_label2.append({"data": [l1, l2], "label": 2})

        # 用完后释放内存
        del flows_with_ge2_le10, flows_with_g10
        if "selected_flows" in locals():
            del selected_flows

        # --- 保存类别2数据 ---
        random.shuffle(samples_label2)
        print(f"保存类别2训练集（{len(samples_label2)} 个样本）...")
        _dump_in_chunks(samples_label2, train_dir, k2, name="class2")
        print(f"类别2训练集保存完成！ 样本数: {len(samples_label2)}")
        count_class2 = len(samples_label2)
        del samples_label2
        gc.collect()

        # from .utils import check_dataset
        # check_dataset(train_dir)

    # ----- 新增逻辑: 生成单条payload测试集，并分片保存 -----
    if k3 <= 0:
        return
    # 准备映射表
    import json
    test_samples = []
    with open(os.path.join(tmp_path, "label_name_to_id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    for label, id_ in label2id.items():
        payloads = execute_sql_on_dbs(dbs, f"SELECT content FROM payloads WHERE label = {id_} ORDER BY RANDOM() LIMIT {k3//len(dbs)+1}", unpack=True)
        for payload in payloads:
            test_samples.append({"data": _str_to_ids(_bigram_generation(payload), seq_length=None, type="bert", CLS_front=True, padding=True), "label": id_})

    random.shuffle(test_samples)
    test_dir = os.path.join(out_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    print(f"保存单条payload测试集（{len(test_samples)} 个样本）到: {test_dir}")
    _dump_in_chunks(test_samples, test_dir, k3, name="test")
    # INSERT_YOUR_CODE
    # 将label2id保存到dest_path的id2label.json
    id2label = {str(v): k for k, v in label2id.items()}
    with open(os.path.join(test_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    del test_samples, payloads
    gc.collect()

#TODO: 未完善
def generate_contrastive_dataset_2(preprocess_path: str, dest_path: str, k1: int = 10000, k2: int = 1000, lines_used: int = 5, num_threads: int = 4):
    import os
    from collections import defaultdict

    def collect_pcap_groups(preprocess_path):
        """
        递归遍历preprocess_path（一级目录为标签，二级目录为流的txt文件），
        收集所有流的txt文件名，并在文件名中检索.pcap，
        用.pcap之前的部分为key，将一样的key对应的文件名放到字典的该key下的列表
        """
        pcap_groups = defaultdict(list)
        txt_labels = defaultdict(list)
        for label in os.listdir(preprocess_path):
            label_dir = os.path.join(preprocess_path, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if not fname.endswith('.txt'):
                    continue
                txt_labels[label].append(os.path.join(label_dir, fname))
                pcap_pos = fname.find('.pcap')
                if pcap_pos != -1:
                    pcap_name = fname[:pcap_pos]
                    pcap_protocol = fname[pcap_pos+6:pcap_pos+9]
                    if pcap_protocol != "TCP" and pcap_protocol != "UDP":
                        print("unknown protocol: ", pcap_protocol)
                        continue
                    pcap_groups[f"{pcap_name}_{pcap_protocol}"].append(os.path.join(label_dir, fname))
        return pcap_groups, txt_labels
    groups, txt_labels = collect_pcap_groups(preprocess_path)

    # INSERT_YOUR_CODE
    for key, value in groups.items():
        print(f"key: {key}, value length: {len(value)}")

    pcap_with_ge2_flows = [(key, len(value)) for key, value in groups.items() if len(value) >= 2]

    import random
    import gc
    import pickle
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    from .utils import _LM_input, _dump_in_chunks
    
    # 并行处理参数配置
    samples_per_thread = k1 // num_threads + (1 if k1 % num_threads > 0 else 0)  # 每个线程处理的样本数
    save_interval = 1000  # 每隔多少个样本存储一次
    
    weights = [n*(n-1) for _, n in pcap_with_ge2_flows]
    
    # 为每个线程生成独立的随机种子
    thread_seeds = [random.randint(0, 2**31 - 1) for _ in range(num_threads)]

    from .utils import _str_to_ids
    system_prompt = """<|im_start|>system
Represent the user's input.<|im_end|> """
    prompt = system_prompt + f"""
<|im_start|>user
<表格开始>"""
    prompt_ids = _str_to_ids(prompt, type="qwen3vl-emb")[0]
    prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
<|endoftext|>"""
    prompt2_ids = _str_to_ids(prompt2, type="qwen3vl-emb")[0]

    def generate_sample(lines):
        nonlocal lines_used
        lines_used_here = lines_used
        sample = _LM_input(lines[:lines_used_here], None, None, [], prompt_ids, prompt2_ids, label=1, extract_payloads_from_lines=True, biased_avoid=True, token_type="qwen3vl-emb")
        while sample["data"][-1].shape[1] > 4096 and lines_used_here > 0:
            lines_used_here -= 2
            sample = _LM_input(lines[:lines_used_here], None, None, [], prompt_ids, prompt2_ids, label=1, extract_payloads_from_lines=True, biased_avoid=True, token_type="qwen3vl-emb")
        if sample["data"][-1].shape[1] > 4096:
            raise Exception(f"样本长度始终大于4096: {lines}")
        return sample
    
    def worker_thread(thread_id, num_samples, start_chunk_idx, seed, pbar):
        """
        工作线程函数，处理指定数量的样本并定期存储
        
        Args:
            thread_id: 线程ID
            num_samples: 该线程需要处理的样本数量
            start_chunk_idx: 该线程存储块的起始编号
            seed: 该线程的随机数种子
            pbar: 共享的tqdm进度条实例
        """
        # 为每个线程创建独立的随机数生成器，避免竞争
        import random
        thread_random = random.Random(seed)
        
        local_samples = []
        chunk_idx = start_chunk_idx
        chunks_written = 0
        
        try:
            for local_idx in range(num_samples):
                # 选择key和文件
                selected_key = thread_random.choices(pcap_with_ge2_flows, weights=weights, k=1)[0][0]
                files = groups[selected_key]
                flow1, flow2 = thread_random.sample(files, 2)
                
                # 读取文件
                with open(flow1, "r", encoding="utf-8") as f:
                    lines1 = f.readlines()[:lines_used]
                with open(flow2, "r", encoding="utf-8") as f:
                    lines2 = f.readlines()[:lines_used]
                
                # 生成样本
                try:
                    sample1 = generate_sample(lines1)
                    sample2 = generate_sample(lines2)
                except Exception as e:
                    import traceback
                    pbar.write(f"生成样本出错: {e}")
                    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    pbar.write(tb_str)
                    continue
                sample = {
                    "data": [sample1["data"], sample2["data"]],
                    "label": 1
                }
                local_samples.append(sample)
                
                # 更新进度条（线程安全）
                pbar.update(1)
                
                # 每隔save_interval个样本存储一次
                if len(local_samples) >= save_interval:
                    # 存储当前批次
                    file_name = f"contrastive_part_{chunk_idx:05d}.pkl"
                    file_path = os.path.join(dest_path, "train", file_name)
                    with open(file_path, "wb") as fout:
                        pickle.dump(local_samples, fout)
                    
                    chunks_written += 1
                    chunk_idx += 1
                    
                    # 清理内存
                    del local_samples
                    local_samples = []
                    gc.collect()
            
            # 处理剩余的样本
            if len(local_samples) > 0:
                file_name = f"contrastive_part_{chunk_idx:05d}.pkl"
                file_path = os.path.join(dest_path, "train", file_name)
                with open(file_path, "wb") as fout:
                    pickle.dump(local_samples, fout)
                chunks_written += 1
                del local_samples
                gc.collect()
        except Exception as e:
            # 如果出错，确保进度条仍然更新以反映已处理的样本
            pbar.refresh()
            raise e
        
        return thread_id, chunks_written
    
    os.makedirs(os.path.join(dest_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(dest_path, "test"), exist_ok=True)
    
    # 计算每个线程的起始块编号，确保不重叠
    # 每个线程分配足够的编号空间，以thread_id作为前缀部分
    max_chunks_per_thread = (samples_per_thread + save_interval - 1) // save_interval + 1  # 向上取整并加1作为缓冲
    start_chunk_indices = [i * max_chunks_per_thread for i in range(num_threads)]
    
    # 创建共享的tqdm进度条
    pbar = tqdm(total=k1, desc="生成对比样本", unit="样本")
    
    # 使用线程池并行处理
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for thread_id in range(num_threads):
                start_idx = thread_id * samples_per_thread
                num_samples = min(samples_per_thread, k1 - start_idx)
                if num_samples <= 0:
                    break
                
                future = executor.submit(
                    worker_thread,
                    thread_id,
                    num_samples,
                    start_chunk_indices[thread_id],
                    thread_seeds[thread_id],
                    pbar
                )
                futures.append(future)
            
            # 等待所有线程完成
            for future in as_completed(futures):
                try:
                    thread_id, chunks_written = future.result()
                    pbar.write(f"线程 {thread_id} 完成，写入 {chunks_written} 个块")
                except Exception as e:
                    pbar.write(f"线程处理出错: {e}")
                    raise
    finally:
        # 确保进度条正确关闭
        pbar.close()

    # 生成测试集
    if k2 > 0:
        print(f"\n开始生成测试集，每个label生成 {k2} 个样本...")
        
        # 构建label到数字的映射（按label名称排序）
        sorted_labels = sorted(txt_labels.keys())
        label2id = {label: idx for idx, label in enumerate(sorted_labels)}
        
        # 计算实际会生成的样本总数（考虑文件数不足的情况）
        total_test_samples = sum(min(len(txt_labels[label]), k2) for label in sorted_labels)
        
        test_samples = []
        test_pbar = tqdm(total=total_test_samples, desc="生成测试集样本", unit="样本")
        
        for label in sorted_labels:
            label_id = label2id[label]
            label_files = txt_labels[label]
            
            # 如果文件数不足k2，则全部使用；否则随机选择k2个
            if len(label_files) < k2:
                selected_files = label_files
                test_pbar.write(f"警告: label '{label}' 只有 {len(label_files)} 个文件，少于要求的 {k2} 个")
            else:
                selected_files = random.sample(label_files, k2)
            
            for file_path in selected_files:
                try:
                    # 读取文件
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    # 生成单个样本（data只包含一条数据，不是列表）
                    sample = generate_sample(lines)
                    
                    # 测试集样本格式：data只包含一条数据，label是标签对应的数字
                    test_sample = {
                        "data": sample["data"],  # 只包含一条数据，不是列表
                        "label": label_id
                    }
                    test_samples.append(test_sample)
                    test_pbar.update(1)
                    
                except Exception as e:
                    import traceback
                    test_pbar.write(f"处理文件 {file_path} 时出错: {e}")
                    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    test_pbar.write(tb_str)
                    continue
        
        test_pbar.close()
        
        # 打乱测试集样本
        random.shuffle(test_samples)
        
        # 保存测试集
        print(f"保存测试集（{len(test_samples)} 个样本）到: {os.path.join(dest_path, "test")}")
        _dump_in_chunks(test_samples, os.path.join(dest_path, "test"), -1, name="test")
        print(f"测试集保存完成！样本数: {len(test_samples)}")
        
        # 保存 id2label.json 到测试目录
        import json
        id2label = {str(idx): label for label, idx in label2id.items()}
        id2label_path = os.path.join(dest_path, "test", "id2label.json")
        with open(id2label_path, "w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False, indent=2)
        print(f"id2label.json 已保存到: {id2label_path}")

def generate_alignment_dataset_1(tmp_path: str, dest_path: str, k1: int = 10000, k2: int = 1000, k3: int = 1000, k4: int = 1000, k5: int = 1000, k6: int = 1000, test_ratio: float = 0.1, 
    packet_num_in_flow: int = 10):
    k1 = max(0, k1)
    k2 = max(0, k2)
    k3 = max(0, k3)
    k4 = max(0, k4)
    k5 = max(0, k5)
    k6 = max(0, k6)
    system_prompt = """<|im_start|>system
你是一个AI助手，擅长阅读表格形式的网络流量并对其进行思考和理解，并能够完成各种针对网络流量的问题。<|im_end|>"""

    """
    从tmp_path随机检索6*k个流，每个流的包数量在3-10之间，使用model.py定义的对多个sqlite进行操作的函数
    """
    import os
    import random
    import gc
    import sys

    from .model import connect_to_dbs, execute_sql_on_dbs, close_dbs
    from .utils import _LM_input, _build_table, _ids_to_str, _str_to_ids, _dump_in_chunks

    train_dir = os.path.join(dest_path, "train")
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(dest_path, "test")
    os.makedirs(test_dir, exist_ok=True)

    conns = connect_to_dbs(tmp_path)
    # 查询每个库中3-10包的流id：先查流id和包数量
    sql = """
    SELECT flows.label, flows.flow_path, flows.flow_type
    FROM flows
    JOIN (
        SELECT flow_id, SUM(packet_count) AS pkt_num
        FROM bursts
        JOIN burst_packet_count ON bursts.id = burst_packet_count.burst_id
        GROUP BY bursts.flow_id
    ) AS pkt_counts
    ON flows.id = pkt_counts.flow_id
    WHERE pkt_counts.pkt_num >= 3
    ORDER BY RANDOM()
    LIMIT {}
    """

    # ----------- 任务1: Table Partition -----------
    if k1 > 0:
        print(f"任务1-生成Table Partition任务样本, 目标数量: {k1}")
        sys.stdout.flush()
        prompt = system_prompt + """
<|im_start|>user
接下来将给出一个表格形式的tcp或udp流量（不包含payload），请输出该流量的第一个单元格、最后一个单元格的元素。
输出时，输出用“,”隔开的答案，不需要输出分隔符，若元素为空则输出“空”。
第一个单元格指第一行、第一列的元素，一般会是表头元素。最后一个元素指最后一行、最后一列的元素。
接下来是流量表格：<表格开始>"""
        prompt_ids = _str_to_ids(prompt, type="qwen3vl")[0]
        prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定表格中第一个单元格、最后一个单元格的元素分别是："""
        prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
        samples1 = []
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0
        candidate_flows = execute_sql_on_dbs(conns, sql.format((k1*2)//len(conns)+1))
        total = len(candidate_flows)
        for idx, (label, flow_path, flow_type) in enumerate(candidate_flows):
            if idx % 100 == 0 or idx == total - 1:
                print(f"[任务1] 进度：{idx+1}/{total}, 有效样本数: {sample_num}/{k1}")
                sys.stdout.flush()
            if sample_num >= k1:
                break
            lines = open(flow_path, "r", encoding="utf-8").readlines()[:packet_num_in_flow]
            try:
                table, payload_bert_ids = _build_table(lines, None, flow_type, extract_payloads_from_lines=False, shuffle_columns=True, random_drop_columns=True, biased_avoid=True)
            except:
                continue
            e_head = _ids_to_str(table[0][0], type="qwen3vl")[1:-1]
            e_tail = _ids_to_str(table[-1][-1], type="qwen3vl")[1:-1]
            if len(e_head) == 0:
                e_head = "空"
            if len(e_tail) == 0:
                e_tail = "空"
            label_text = f"{e_head},{e_tail}<|im_end|>"
            label_ids = _str_to_ids(label_text, type="qwen3vl")[0]
            sample = _LM_input(None, None, flow_type, label_ids, prompt_ids, prompt2_ids, label=1, _build_table_result=(table, payload_bert_ids))
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples1.append(sample)
            sample_num += 1
            if len(samples1) >= batch_size:
                random.shuffle(samples1)
                n_train = int(len(samples1) * (1 - test_ratio))
                train_samples1, test_samples1 = samples1[:n_train], samples1[n_train:]
                _dump_in_chunks(train_samples1, train_dir, -1, name=f"task1_train_{train_batch_idx}")
                _dump_in_chunks(test_samples1, test_dir, -1, name=f"task1_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples1, test_samples1
                samples1 = []
                gc.collect()
        # 处理剩余样本
        if len(samples1) > 0:
            random.shuffle(samples1)
            n_train = int(len(samples1) * (1 - test_ratio))
            train_samples1, test_samples1 = samples1[:n_train], samples1[n_train:]
            _dump_in_chunks(train_samples1, train_dir, -1, name=f"task1_train_{train_batch_idx}")
            _dump_in_chunks(test_samples1, test_dir, -1, name=f"task1_test_{test_batch_idx}")
            del train_samples1, test_samples1, samples1
        del candidate_flows
        gc.collect()

    # ----------- 任务2: Cell LookUp -----------
    if k2 > 0:
        print(f"任务2-生成Cell LookUp任务样本, 目标数量: {k2}")
        sys.stdout.flush()
        prompt = system_prompt + """
<|im_start|>user
接下来将给出一个表格形式的tcp或udp流量（不包含payload），以及一个元素的值，请你在表格中找到该元素并输出其所在的行和列的序号。
输出时，行和列的序号仅用“|”隔开，序号从1开始；若有多个单元格符合条件，请输出所有符合条件的单元格的行和列序号，用“,”隔开。
保证至少有一个单元格符合条件，且元素的值非空。
元素的值是“{}”。
接下来是流量表格：<表格开始>"""
        prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定表格中符合条件的单元格的行和列的序号分别是："""
        prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
        empty_str = "<>"
        empty_str_ids = _str_to_ids(empty_str, type="qwen3vl")[0]
        samples2 = []
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0
        candidate_flows = execute_sql_on_dbs(conns, sql.format((k2*2)//len(conns)+1))
        total = len(candidate_flows)
        for idx, (label, flow_path, flow_type) in enumerate(candidate_flows):
            if idx % 100 == 0 or idx == total - 1:
                print(f"[任务2] 进度：{idx+1}/{total}, 有效样本数: {sample_num}/{k2}")
                sys.stdout.flush()
            if sample_num >= k2:
                break
            lines = open(flow_path, "r", encoding="utf-8").readlines()[:packet_num_in_flow]
            try:
                table, payload_bert_ids = _build_table(lines, None, flow_type, extract_payloads_from_lines=False, shuffle_columns=True, random_drop_columns=True, biased_avoid=True)
            except:
                continue
            e = empty_str_ids
            column_num = len(table)
            row_num = len(table[0])
            def cmp(e1, e2):
                return all(ee == eee for ee, eee in zip(e1, e2))
            while cmp(e, empty_str_ids):
                column = random.randint(0, column_num-1)
                row = random.randint(0, row_num-1)
                e = table[column][row]
            e_str = _ids_to_str(e, type="qwen3vl")[1:-1]
            prompt_ = prompt.format(e_str)
            prompt_ids = _str_to_ids(prompt_, type="qwen3vl")[0]
            label_text = []
            for i in range(column_num):
                for j in range(row_num):
                    if cmp(e, table[i][j]):
                        label_text.append(f"{j+1}|{i+1}")
            label_text = ",".join(label_text) + "<|im_end|>"
            label_ids = _str_to_ids(label_text, type="qwen3vl")[0]
            sample = _LM_input(None, None, flow_type, label_ids, prompt_ids, prompt2_ids, label=2, _build_table_result=(table, payload_bert_ids))
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples2.append(sample)
            sample_num += 1
            if len(samples2) >= batch_size:
                random.shuffle(samples2)
                n_train = int(len(samples2) * (1 - test_ratio))
                train_samples2, test_samples2 = samples2[:n_train], samples2[n_train:]
                _dump_in_chunks(train_samples2, train_dir, -1, name=f"task2_train_{train_batch_idx}")
                _dump_in_chunks(test_samples2, test_dir, -1, name=f"task2_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples2, test_samples2
                samples2 = []
                gc.collect()
        # 处理剩余样本
        if len(samples2) > 0:
            random.shuffle(samples2)
            n_train = int(len(samples2) * (1 - test_ratio))
            train_samples2, test_samples2 = samples2[:n_train], samples2[n_train:]
            _dump_in_chunks(train_samples2, train_dir, -1, name=f"task2_train_{train_batch_idx}")
            _dump_in_chunks(test_samples2, test_dir, -1, name=f"task2_test_{test_batch_idx}")
            del train_samples2, test_samples2, samples2
        del candidate_flows
        gc.collect()

    # ----------- 任务3: Reverse LookUp -----------
    if k3 > 0:
        print(f"任务3-生成Reverse LookUp任务样本, 目标数量: {k3}")
        sys.stdout.flush()
        prompt = system_prompt + """
<|im_start|>user
接下来将给出一个表格形式的tcp或udp流量（不包含payload），以及一组行和列的序号，请你在表格中找到对应的单元格并输出其值。
输出时，仅输出答案；若对应单元格的值为空，则输出"空"。
保证行和列的序号都在表格范围内。
行和列的序号是：{}。
接下来是流量表格：<表格开始>"""
        prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定表格中符合条件的单元格的值是："""
        prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
        samples3 = []
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0
        candidate_flows = execute_sql_on_dbs(conns, sql.format((k3*2)//len(conns)+1))
        total = len(candidate_flows)
        for idx, (label, flow_path, flow_type) in enumerate(candidate_flows):
            if idx % 100 == 0 or idx == total - 1:
                print(f"[任务3] 进度：{idx+1}/{total}, 有效样本数: {sample_num}/{k3}")
                sys.stdout.flush()
            if sample_num >= k3:
                break
            lines = open(flow_path, "r", encoding="utf-8").readlines()[:packet_num_in_flow]
            try:
                table, payload_bert_ids = _build_table(lines, None, flow_type, extract_payloads_from_lines=False, shuffle_columns=True, random_drop_columns=True, biased_avoid=True)
            except:
                continue
            column_num = len(table)
            row_num = len(table[0])
            column = random.randint(0, column_num-1)
            row = random.randint(0, row_num-1)
            prompt_ = prompt.format(f"{row+1}|{column+1}")
            prompt_ids = _str_to_ids(prompt_, type="qwen3vl")[0]
            e = table[column][row]
            label_text = _ids_to_str(e, type="qwen3vl")[1:-1]
            if len(label_text) == 0:
                label_text = "空"
            label_text = label_text + "<|im_end|>"
            label_ids = _str_to_ids(label_text, type="qwen3vl")[0]
            sample = _LM_input(None, None, flow_type, label_ids, prompt_ids, prompt2_ids, label=3, _build_table_result=(table, payload_bert_ids))
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples3.append(sample)
            sample_num += 1
            if len(samples3) >= batch_size:
                random.shuffle(samples3)
                n_train = int(len(samples3) * (1 - test_ratio))
                train_samples3, test_samples3 = samples3[:n_train], samples3[n_train:]
                _dump_in_chunks(train_samples3, train_dir, -1, name=f"task3_train_{train_batch_idx}")
                _dump_in_chunks(test_samples3, test_dir, -1, name=f"task3_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples3, test_samples3
                samples3 = []
                gc.collect()
        # 处理剩余样本
        if len(samples3) > 0:
            random.shuffle(samples3)
            n_train = int(len(samples3) * (1 - test_ratio))
            train_samples3, test_samples3 = samples3[:n_train], samples3[n_train:]
            _dump_in_chunks(train_samples3, train_dir, -1, name=f"task3_train_{train_batch_idx}")
            _dump_in_chunks(test_samples3, test_dir, -1, name=f"task3_test_{test_batch_idx}")
            del train_samples3, test_samples3, samples3
        del candidate_flows
        gc.collect()

    # ----------- 任务4: Column Retrieval -----------
    if k4 > 0:
        print(f"任务4-生成Column Retrieval任务样本, 目标数量: {k4}")
        sys.stdout.flush()
        prompt = system_prompt + """
<|im_start|>user
接下来将给出一个表格形式的tcp或udp流量（不包含payload），以及一个列的序号，请你在表格中找到对应的列并输出其列名称。
输出时，仅输出答案。
保证列的序号都在表格范围内；保证列名称不为空。
列的序号是：{}。
接下来是流量表格：<表格开始>"""
        prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定表格中对应列的名称是："""
        prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
        samples4 = []
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0
        candidate_flows = execute_sql_on_dbs(conns, sql.format((k4*2)//len(conns)+1))
        total = len(candidate_flows)
        for idx, (label, flow_path, flow_type) in enumerate(candidate_flows):
            if idx % 100 == 0 or idx == total - 1:
                print(f"[任务4] 进度：{idx+1}/{total}, 有效样本数: {sample_num}/{k4}")
                sys.stdout.flush()
            if sample_num >= k4:
                break
            lines = open(flow_path, "r", encoding="utf-8").readlines()[:packet_num_in_flow]
            try:
                table, payload_bert_ids = _build_table(lines, None, flow_type, extract_payloads_from_lines=False, shuffle_columns=True, random_drop_columns=True, biased_avoid=True)
            except:
                continue
            column_num = len(table)
            column = random.randint(0, column_num-1)
            column_name = _ids_to_str(table[column][0], type="qwen3vl")[1:-1]
            prompt_ = prompt.format(str(column+1))
            prompt_ids = _str_to_ids(prompt_, type="qwen3vl")[0]
            label_text = column_name + "<|im_end|>"
            label_ids = _str_to_ids(label_text, type="qwen3vl")[0]
            sample = _LM_input(None, None, flow_type, label_ids, prompt_ids, prompt2_ids, label=4, _build_table_result=(table, payload_bert_ids))
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples4.append(sample)
            sample_num += 1
            if len(samples4) >= batch_size:
                random.shuffle(samples4)
                n_train = int(len(samples4) * (1 - test_ratio))
                train_samples4, test_samples4 = samples4[:n_train], samples4[n_train:]
                _dump_in_chunks(train_samples4, train_dir, -1, name=f"task4_train_{train_batch_idx}")
                _dump_in_chunks(test_samples4, test_dir, -1, name=f"task4_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples4, test_samples4
                samples4 = []
                gc.collect()
        # 处理剩余样本
        if len(samples4) > 0:
            random.shuffle(samples4)
            n_train = int(len(samples4) * (1 - test_ratio))
            train_samples4, test_samples4 = samples4[:n_train], samples4[n_train:]
            _dump_in_chunks(train_samples4, train_dir, -1, name=f"task4_train_{train_batch_idx}")
            _dump_in_chunks(test_samples4, test_dir, -1, name=f"task4_test_{test_batch_idx}")
            del train_samples4, test_samples4, samples4
        del candidate_flows
        gc.collect()    

    # ----------- 任务5: Row Retrieval -----------
    if k5 > 0:
        print(f"任务5-生成Row Retrieval任务样本, 目标数量: {k5}")
        sys.stdout.flush()
        prompt = system_prompt + """
<|im_start|>user
接下来将给出一个表格形式的tcp或udp流量（不包含payload），以及一个行的序号，请你在表格中找到对应的行并输出所有元素。
输出时，元素之间用“,”隔开；若元素为空，则输出“空”。
保证行的序号都在表格范围内。
行的序号是：{}。
接下来是流量表格：<表格开始>"""
        prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定表格中对应行的所有元素是："""
        prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
        samples5 = []
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0
        candidate_flows = execute_sql_on_dbs(conns, sql.format((k5*2)//len(conns)+1))
        total = len(candidate_flows)
        for idx, (label, flow_path, flow_type) in enumerate(candidate_flows):
            if idx % 100 == 0 or idx == total - 1:
                print(f"[任务5] 进度：{idx+1}/{total}, 有效样本数: {sample_num}/{k5}")
                sys.stdout.flush()
            if sample_num >= k5:
                break
            lines = open(flow_path, "r", encoding="utf-8").readlines()[:packet_num_in_flow]
            try:
                table, payload_bert_ids = _build_table(lines, None, flow_type, extract_payloads_from_lines=False, shuffle_columns=True, random_drop_columns=True, biased_avoid=True)
            except:
                continue
            row_num = len(table[0])
            row = random.randint(0, row_num-1)
            prompt_ = prompt.format(f"{row+1}")
            prompt_ids = _str_to_ids(prompt_, type="qwen3vl")[0]
            label_text = []
            for column in table:
                e = column[row]
                e_str = _ids_to_str(e, type="qwen3vl")[1:-1]
                if len(e_str) == 0:
                    e_str = "空"
                label_text.append(e_str)
            label_text = ",".join(label_text) + "<|im_end|>"
            label_ids = _str_to_ids(label_text, type="qwen3vl")[0]
            sample = _LM_input(None, None, flow_type, label_ids, prompt_ids, prompt2_ids, label=5, _build_table_result=(table, payload_bert_ids))
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples5.append(sample)
            sample_num += 1
            if len(samples5) >= batch_size:
                random.shuffle(samples5)
                n_train = int(len(samples5) * (1 - test_ratio))
                train_samples5, test_samples5 = samples5[:n_train], samples5[n_train:]
                _dump_in_chunks(train_samples5, train_dir, -1, name=f"task5_train_{train_batch_idx}")
                _dump_in_chunks(test_samples5, test_dir, -1, name=f"task5_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples5, test_samples5
                samples5 = []
                gc.collect()
        # 处理剩余样本
        if len(samples5) > 0:
            random.shuffle(samples5)
            n_train = int(len(samples5) * (1 - test_ratio))
            train_samples5, test_samples5 = samples5[:n_train], samples5[n_train:]
            _dump_in_chunks(train_samples5, train_dir, -1, name=f"task5_train_{train_batch_idx}")
            _dump_in_chunks(test_samples5, test_dir, -1, name=f"task5_test_{test_batch_idx}")
            del train_samples5, test_samples5, samples5
        del candidate_flows
        gc.collect()

    # ----------- 任务6: Size Detection -----------
    if k6 > 0:
        print(f"任务6-生成Size Detection任务样本, 目标数量: {k6}")
        sys.stdout.flush()
        prompt = system_prompt + """
<|im_start|>user
接下来将给出一个表格形式的tcp或udp流量（不包含payload），请输出该表的行数和列数。
输出时，仅输出答案，行数和列数之间用“,”隔开。
保证行数和列数都不为0。
接下来是流量表格：<表格开始>"""
        prompt_ids = _str_to_ids(prompt, type="qwen3vl")[0]
        prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定表格中行数和列数分别是："""
        prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
        samples6 = []
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0
        candidate_flows = execute_sql_on_dbs(conns, sql.format((k6*2)//len(conns)+1))
        total = len(candidate_flows)
        for idx, (label, flow_path, flow_type) in enumerate(candidate_flows):
            if idx % 100 == 0 or idx == total - 1:
                print(f"[任务6] 进度：{idx+1}/{total}, 有效样本数: {sample_num}/{k6}")
                sys.stdout.flush()
            if sample_num >= k6:
                break
            lines = open(flow_path, "r", encoding="utf-8").readlines()[:packet_num_in_flow]
            try:
                table, payload_bert_ids = _build_table(lines, None, flow_type, extract_payloads_from_lines=False, shuffle_columns=True, random_drop_columns=True, biased_avoid=True)
            except:
                continue
            column_num = len(table)
            row_num = len(table[0])
            label_text = f"{row_num},{column_num}<|im_end|>"
            label_ids = _str_to_ids(label_text, type="qwen3vl")[0]
            sample = _LM_input(None, None, flow_type, label_ids, prompt_ids, prompt2_ids, label=6, _build_table_result=(table, payload_bert_ids))
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples6.append(sample)
            sample_num += 1
            if len(samples6) >= batch_size:
                random.shuffle(samples6)
                n_train = int(len(samples6) * (1 - test_ratio))
                train_samples6, test_samples6 = samples6[:n_train], samples6[n_train:]
                _dump_in_chunks(train_samples6, train_dir, -1, name=f"task6_train_{train_batch_idx}")
                _dump_in_chunks(test_samples6, test_dir, -1, name=f"task6_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples6, test_samples6
                samples6 = []
                gc.collect()
        # 处理剩余样本
        if len(samples6) > 0:
            random.shuffle(samples6)
            n_train = int(len(samples6) * (1 - test_ratio))
            train_samples6, test_samples6 = samples6[:n_train], samples6[n_train:]
            _dump_in_chunks(train_samples6, train_dir, -1, name=f"task6_train_{train_batch_idx}")
            _dump_in_chunks(test_samples6, test_dir, -1, name=f"task6_test_{test_batch_idx}")
            del train_samples6, test_samples6, samples6
        del candidate_flows
        gc.collect()


def generate_alignment_dataset_2(
    tmp_path: str, dest_path: str,
    k1: int = 10000, k2: int = 1000, k3: int = 1000, k4: int = 1000, FIRST_BYTES: int = 8,
    test_ratio: float = 0.1, burst_packet_num: int = 10
):
    """
    读取generate_classify_tmp、process_flow_dataset生成的中间数据，分别生成如下四类任务的数据并划分训练集/测试集：
    - 每个样本是一个二维列表和标签的元组：(二维列表, 自然语言标签)
    - 二维列表的第一行是表头，最后一列放payload
    - 标签为自然语言句式、题目描述式生成
    """
    print("=== 开始生成齐纳对齐数据集(generate_alignment_dataset) ===")
    import os
    import pickle
    import random
    import json
    import gc
    import sys
    import torch

    # INSERT_YOUR_CODE
    from .model import connect_to_dbs, execute_sql_on_dbs, close_dbs
    dbs = connect_to_dbs(tmp_path)

    from .utils import _str_to_ids, _build_table, _dump_in_chunks, _LM_input

    train_dir = os.path.join(dest_path, "train")
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(dest_path, "test")
    os.makedirs(test_dir, exist_ok=True)

    system_prompt = """<|im_start|>system
你是一个AI助手，擅长阅读表格形式的网络流量并对其进行思考和理解，并能够完成各种针对网络流量的问题。<|im_end|>"""

    # ----------- 1. payload属性挖掘 -----------
    if k1 > 0:
        print(f"任务1-采样payload并生成属性挖掘任务样本, 目标数量: {k1}")
        samples1 = []
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0
        selected_payloads = execute_sql_on_dbs(dbs, f"SELECT content, flow_type FROM payloads ORDER BY RANDOM() LIMIT {(k1*2)//len(dbs)+1}")
        sample_type = 0
        total = len(selected_payloads)
        prompt = system_prompt + f"""
<|im_start|>user
对于接下来给定的一个tcp或udp的payload，请输出该payload的类型（TCP或UDP）、长度、前{FIRST_BYTES}字节。
输出时，元素之间用“,”隔开。
接下来是payload的内容：<|image_pad|><|im_end|>
<|im_start|>assistant
给定payload的类型、长度和前{FIRST_BYTES}字节分别是："""
        prompt_ids = _str_to_ids(prompt, type="qwen3vl")[0]
        print(f"实际获得payload样本数: {total}")
        for idx, (payload, flow_type) in enumerate(selected_payloads):
            if idx % max(1, total // 10) == 0:
                print(f"任务1处理进度: {idx}/{total}, 有效样本数: {sample_num}/{k1}")
                sys.stdout.flush()
            if sample_num >= k1:
                break
            payload_prefix = payload[:FIRST_BYTES]
            length = len(payload)
            label_text = f"{flow_type}，{length}，{payload_prefix}<|im_end|>"
            payload_ids, valid_len = _str_to_ids(payload, 1500, "bert", True, True)
            attention_mask_payload = [1]*valid_len+[0]*(1500-valid_len)
            global_attention_mask_payload = [0]*1500
            global_attention_mask_payload[0] = 1
            payload = (payload_ids, attention_mask_payload, global_attention_mask_payload)
            label_ids = _str_to_ids(label_text, type="qwen3vl")[0]
            position_ids = torch.arange(len(prompt_ids)+len(label_ids)).unsqueeze(0).expand(3, -1)
            sample = {
                "data": (prompt_ids, label_ids, [payload], position_ids),
                "label": 1
            }
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples1.append(sample)
            sample_num += 1
            if len(samples1) >= batch_size:
                random.shuffle(samples1)
                n_train = int(len(samples1) * (1 - test_ratio))
                train_samples1, test_samples1 = samples1[:n_train], samples1[n_train:]
                _dump_in_chunks(train_samples1, train_dir, -1, name=f"task1_train_{train_batch_idx}")
                _dump_in_chunks(test_samples1, test_dir, -1, name=f"task1_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples1, test_samples1
                samples1 = []
                gc.collect()
        # 处理剩余样本
        if len(samples1) > 0:
            random.shuffle(samples1)
            n_train = int(len(samples1) * (1 - test_ratio))
            train_samples1, test_samples1 = samples1[:n_train], samples1[n_train:]
            _dump_in_chunks(train_samples1, train_dir, -1, name=f"task1_train_{train_batch_idx}")
            _dump_in_chunks(test_samples1, test_dir, -1, name=f"task1_test_{test_batch_idx}")
            del train_samples1, test_samples1, samples1
        print("任务1-属性挖掘样本处理完成。")
        del selected_payloads
        gc.collect()
    
    # ----------- 2. 属性和payload的编号对应关系 -----------
    # 先随机选择k2个payload样本，再按group_size进行分组（每组group_size个payload），k2/group_size即为组数
    if k2 > 0:
        print("任务2-生成属性和payload的编号对应关系任务...")
        """
        读取generate_classify_tmp、process_flow_dataset生成的中间数据，分别生成如下四类任务的数据并划分训练集/测试集：
        - 每个样本是一个二维列表和标签的元组：(二维列表, 自然语言标签)
        - 二维列表的第一行是表头，最后一列放payload
        - 标签为自然语言句式、题目描述式生成
        """
        group_size = 4
        num_groups = k2 // group_size
        total_payload_needed = num_groups * group_size * 2  # 增加采样数量

        # 随机从payloads表中选择k2*5行并join packets表获取line字段
        # 查询k2*5个payload和其对应的packet.line
        payload_packet_rows = execute_sql_on_dbs(dbs, f"""
            SELECT payloads.content, packets.line, payloads.flow_type
            FROM payloads
            JOIN packets ON payloads.packet_id = packets.id
            ORDER BY RANDOM()
            LIMIT {total_payload_needed//len(dbs)+1}
        """)
        # 将payload_packet_rows按flow_type划分成两个列表
        from collections import defaultdict
        type_to_rows = defaultdict(list)
        for row in payload_packet_rows:
            flow_type = row[2]
            type_to_rows[flow_type].append(row)
        # 再对每个列表都按group_size一组划分为样本，并且所有样本都放在all_grouped_samples中
        all_grouped_samples = []
        for rows in type_to_rows.values():
            num_available = (len(rows) // group_size) * group_size
            for i in range(0, num_available, group_size):
                group = rows[i:i+group_size]
                if len(group) == group_size:
                    all_grouped_samples.append(group)

        samples2 = []
        sample_type = 1
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0

        prompt = system_prompt + f"""
<|im_start|>user
接下来会给出一个表格，包含{group_size}个不一定相关的包的头部特征和统计特征，以及在最后一列的打乱顺序的这些包的payload，请输出每个payload原有的顺序编号。
输出时，元素之间用“,”隔开。
保证这些包的payload非空。
接下来是表格：<表格开始>"""
        prompt_ids = _str_to_ids(prompt, type="qwen3vl")[0]
        prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定表格中每个payload原有的顺序编号是："""
        prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
        random.shuffle(all_grouped_samples)
        for group in all_grouped_samples:
            if sample_num >= k2:
                break
            payload_contents = [row[0] for row in group]
            lines = [row[1] for row in group]
            flow_type = group[0][2]
            indices = list(range(group_size))
            random.shuffle(indices)
            shuffled_payload_contents = [payload_contents[i] for i in indices]
            order_text = ",".join(str(i + 1) for i in indices)
            label_ids = _str_to_ids(order_text+"<|im_end|>", type="qwen3vl", CLS_front=False, seq_length=50)[0]
            try:
                sample = _LM_input(lines, shuffled_payload_contents, flow_type, label_ids, prompt_ids, prompt2_ids, label=2, extract_payloads_from_lines=False, biased_avoid=True)
            except:
                continue
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples2.append(sample)
            sample_num += group_size
            if len(samples2) >= batch_size:
                random.shuffle(samples2)
                n_train = int(len(samples2) * (1 - test_ratio))
                train_samples2, test_samples2 = samples2[:n_train], samples2[n_train:]
                _dump_in_chunks(train_samples2, train_dir, -1, name=f"task2_train_{train_batch_idx}")
                _dump_in_chunks(test_samples2, test_dir, -1, name=f"task2_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples2, test_samples2
                samples2 = []
                gc.collect()
        # 处理剩余样本
        if len(samples2) > 0:
            random.shuffle(samples2)
            n_train = int(len(samples2) * (1 - test_ratio))
            train_samples2, test_samples2 = samples2[:n_train], samples2[n_train:]
            _dump_in_chunks(train_samples2, train_dir, -1, name=f"task2_train_{train_batch_idx}")
            _dump_in_chunks(test_samples2, test_dir, -1, name=f"task2_test_{test_batch_idx}")
            del train_samples2, test_samples2, samples2
        del payload_packet_rows
        gc.collect()
    
    # ----------- 3. payload排序 -----------
    # 选取burst的payload序列，将非空的payload排在前面的位置、打乱顺序，输出非空的payload的编号排列
    if k3 > 0:
        print("任务3-生成payload排序任务...")
        samples3 = []
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0
        # 从conn的bursts表中随机抽取k3*5个burst的payload序列
        selected_bursts = execute_sql_on_dbs(
            dbs,
            f"""
            SELECT id, flow_type
            FROM bursts
            JOIN burst_packet_count ON bursts.id = burst_packet_count.burst_id
            WHERE bursts.payload_num >= 3 AND burst_packet_count.packet_count <= {burst_packet_num}
            ORDER BY RANDOM()
            LIMIT {(k3*2)//len(dbs)+1}
            """
        )
        from tqdm import tqdm
        print("任务3-生成样本中...")
        prompt = system_prompt + """
<|im_start|>user
接下来会给出一个表格，包含一个burst的若干个包的非空payload的打乱顺序的序列。请输出非空payload在原始序列中的编号排列。
输出时，元素之间用“,”隔开。
该burst中非空payload数量为{}， 空payload数量为：{}。
接下来是表格：<表格开始>"""
        prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定表格中非空payload在原始序列中的编号排列是："""
        prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
        for idx, (burst_id, flow_type) in enumerate(selected_bursts):
            if idx % 100 == 0:
                print(f"任务3进度: {idx}/{len(selected_bursts)}, 有效样本数: {sample_num}/{k3}")
                sys.stdout.flush()
            if sample_num >= k3:
                break
            # 从conn的payloads表中获取burst的payload序列
            # 从packets表选择burst_id=burst_id的payload_id，join payloads表获取content，若payload_id为None则也是None
            payloads = execute_sql_on_dbs(dbs, f"""
                SELECT payloads.content
                FROM packets
                LEFT JOIN payloads ON packets.payload_id = payloads.id
                WHERE packets.burst_id = {burst_id}
                ORDER BY packets.id
            """, unpack=True)
            non_empty_payloads = [(i, pl) for i, pl in enumerate(payloads) if pl is not None and pl.strip()]
            prompt_ = prompt.format(len(non_empty_payloads), len(payloads)-len(non_empty_payloads))
            prompt_ids = _str_to_ids(prompt_, type="qwen3vl")[0]
            indices = list(range(len(non_empty_payloads)))
            random.shuffle(indices)
            shuffled_payloads = [non_empty_payloads[i][1] for i in indices]
            order_text = ",".join(str(non_empty_payloads[i][0] + 1) for i in indices) + "<|im_end|>"
            label_ids = _str_to_ids(order_text, type="qwen3vl")[0]
            try:
                sample = _LM_input(None, shuffled_payloads, flow_type, label_ids, prompt_ids, prompt2_ids, label=3, extract_payloads_from_lines=False, biased_avoid=True)
            except:
                continue
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples3.append(sample)
            sample_num += 1
            if len(samples3) >= batch_size:
                random.shuffle(samples3)
                n_train = int(len(samples3) * (1 - test_ratio))
                train_samples3, test_samples3 = samples3[:n_train], samples3[n_train:]
                _dump_in_chunks(train_samples3, train_dir, -1, name=f"task3_train_{train_batch_idx}")
                _dump_in_chunks(test_samples3, test_dir, -1, name=f"task3_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples3, test_samples3
                samples3 = []
                gc.collect()
        # 处理剩余样本
        if len(samples3) > 0:
            random.shuffle(samples3)
            n_train = int(len(samples3) * (1 - test_ratio))
            train_samples3, test_samples3 = samples3[:n_train], samples3[n_train:]
            _dump_in_chunks(train_samples3, train_dir, -1, name=f"task3_train_{train_batch_idx}")
            _dump_in_chunks(test_samples3, test_dir, -1, name=f"task3_test_{test_batch_idx}")
            del train_samples3, test_samples3, samples3
        del selected_bursts
        gc.collect()

    # ----------- 4. payload排序+统计特征对齐 -----------
    # 类似3，但额外给出header/统计特征顺序化表达，要求也输出payload正确排序
    if k4 > 0:
        print("任务4-生成payload排序+统计特征对齐任务...")
        selected_bursts = execute_sql_on_dbs(
            dbs,
            f"""
            SELECT id, flow_type
            FROM bursts
            JOIN burst_packet_count ON bursts.id = burst_packet_count.burst_id
            WHERE bursts.payload_num >= 3 AND burst_packet_count.packet_count <= {burst_packet_num}
            ORDER BY RANDOM()
            LIMIT {(k4*2)//len(dbs)+1}
            """
        )
        samples4 = []
        batch_size = 10000
        train_batch_idx = 0
        test_batch_idx = 0
        sample_num = 0
        # 仿照任务1，直接打印百分比进度
        total4 = len(selected_bursts)
        prompt = system_prompt + """
<|im_start|>user
接下来会给出一个流量表格，包含一个burst的若干个包的头部特征和统计特征，以及在最后一列的打乱顺序的这些包的payload。其中非空payload排在前面，空payload排在后面。请输出非空payload原有的顺序编号。
输出时，元素之间用“,”隔开。
该表格中非空payload数量为{}， 空payload数量为：{}。
接下来是表格：<表格开始> """
        prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定表格中每个非空payload原有的顺序编号是："""
        prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
        for i, (burst_id, flow_type) in enumerate(selected_bursts):
            if (i+1) % max(1, total4//20) == 0 or i == 0 or (i+1)==total4:
                print(f"任务4: 生成进度 {i+1}/{total4} ({(i+1)/total4:.1%}), 有效样本数: {sample_num}/{k4}")
                sys.stdout.flush()
            if sample_num >= k4:
                break
            # 从conn的payloads表中获取burst的payload序列
            rows = execute_sql_on_dbs(dbs, f"""
                SELECT packets.line, payloads.content
                FROM packets
                LEFT JOIN payloads ON packets.payload_id = payloads.id
                WHERE packets.burst_id = {burst_id}
                ORDER BY packets.id
            """)
            lines = [row[0] for row in rows]
            payloads = [row[1] for row in rows]
            payloads = [(i, pl) for i, pl in enumerate(payloads) if pl is not None and pl.strip()]
            prompt_ = prompt.format(len(payloads), len(lines)-len(payloads))
            prompt_ids = _str_to_ids(prompt_, type="qwen3vl")[0]
            indices = list(range(len(payloads)))
            random.shuffle(indices)
            shuffled_payloads = [payloads[i][1] for i in indices]
            shuffled_payloads.extend(["" for i in range(len(payloads), len(lines))])
            order_text = ",".join(str(payloads[i][0] + 1) for i in indices)+"<|im_end|>"
            label_ids = _str_to_ids(order_text, type="qwen3vl")[0]
            try:
                sample = _LM_input(lines, shuffled_payloads, flow_type, label_ids, prompt_ids, prompt2_ids, label=4, extract_payloads_from_lines=False, biased_avoid=True)
            except:
                continue
            if sample["data"][-1].shape[1] > 4096:
                continue
            samples4.append(sample)
            sample_num += 1
            if len(samples4) >= batch_size:
                random.shuffle(samples4)
                n_train = int(len(samples4) * (1 - test_ratio))
                train_samples4, test_samples4 = samples4[:n_train], samples4[n_train:]
                _dump_in_chunks(train_samples4, train_dir, -1, name=f"task4_train_{train_batch_idx}")
                _dump_in_chunks(test_samples4, test_dir, -1, name=f"task4_test_{test_batch_idx}")
                train_batch_idx += 1
                test_batch_idx += 1
                del train_samples4, test_samples4
                samples4 = []
                gc.collect()
        # 处理剩余样本
        if len(samples4) > 0:
            random.shuffle(samples4)
            n_train = int(len(samples4) * (1 - test_ratio))
            train_samples4, test_samples4 = samples4[:n_train], samples4[n_train:]
            _dump_in_chunks(train_samples4, train_dir, -1, name=f"task4_train_{train_batch_idx}")
            _dump_in_chunks(test_samples4, test_dir, -1, name=f"task4_test_{test_batch_idx}")
            del train_samples4, test_samples4, samples4
        del selected_bursts
        gc.collect()
    close_dbs(dbs)

def generate_finetuning_catalog(preprocess_path: str, dest_path: str, k: int = 500):
    """
    从preprocess_path中对每个label读取k个文件，按照和generate_finetuning_dataset一样的逻辑生成原始pcap文件名称的记录。
    每个label的目录下生成三个txt文件（train.txt, val.txt, test.txt），比例为8:1:1。
    每个pcap名称占一行，每个label处理完毕后整理内存。

    Args:
        preprocess_path (str): 预处理文件的根目录，目录结构: preprocess_path/label_name/*.txt
        dest_path (str): 保存catalog的目的地目录，每个label会在dest_path/label_name/下生成train.txt, val.txt, test.txt
        k (int): 每个标签最多采集的文件数量
        packet_num_in_flow (int): 每个流包含的包数量
    """
    import os
    import random
    import shutil
    from .utils import _LM_input, _str_to_ids
    from tqdm import tqdm
    import sys
    import gc
    
    # 确保目标目录存在
    os.makedirs(dest_path, exist_ok=True)

    # 获取所有label子目录（必须是目录）
    label_names = [name for name in os.listdir(preprocess_path)
                   if os.path.isdir(os.path.join(preprocess_path, name))]

    # 准备prompt（和generate_finetuning_dataset一样的逻辑）
    system_prompt = """<|im_start|>system
你是一个AI助手，擅长阅读表格形式的网络流量并对其进行思考和理解，并能够完成各种针对网络流量的问题。<|im_end|> """
    prompt = system_prompt + f"""
<|im_start|>user
接下来会给出一个流量表格，包含若干个包的头部特征和统计特征，以及在最后一列的payload。请输出对应的类别。
类别包含: {", ".join(label_names)}。
接下来是表格：<表格开始>"""
    prompt_ids = _str_to_ids(prompt, type="qwen3vl")[0]
    prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定流的类别是："""
    prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
    
    for label in label_names:
        label_ids = _str_to_ids(label+"<|im_end|>", type="qwen3vl")[0]
        label_dir = os.path.join(preprocess_path, label)
        if not os.path.isdir(label_dir):
            continue
        
        # 收集.txt文件
        file_list = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        if not file_list:
            continue
        
        # 随机采样k*5个文件（和generate_finetuning_dataset一样的逻辑）
        if len(file_list) < 10:
            continue
        random.shuffle(file_list)
        sampled_files = file_list
        
        # 收集有效的pcap文件名
        pcap_names = []
        sample_num = 0
        
        for filename in tqdm(sampled_files, desc=f"处理{label}文件", file=sys.stdout):
            # 按照和generate_finetuning_dataset一样的逻辑检查文件
            lines = open(os.path.join(label_dir, filename), "r", encoding="utf-8").readlines()
            if len(lines) < 3:
                continue
            # lines = lines[:packet_num_in_flow]
            # try:
            #     sample = _LM_input(lines, None, None, label_ids, prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True)
            # except:
            #     continue
            # if sample["data"][-1].shape[1] > 4096:
            #     continue
            
            # 提取pcap文件名（和generate_finetuning_dataset一样的逻辑）
            base = filename.rsplit('.', 1)[0]
            pcapname = base + ".pcap"
            pcap_names.append(pcapname)
            sample_num += 1
            
            if sample_num >= k:
                break

        if len(pcap_names) < 10:
            continue

        # 按8:1:1比例划分训练集、验证集、测试集
        random.shuffle(pcap_names)  # 打乱顺序
        total = len(pcap_names)
        num_train = int(total * 0.8)
        num_val = int(total * 0.1)
        # 剩余的是测试集
        
        train_pcaps = pcap_names[:num_train]
        val_pcaps = pcap_names[num_train:num_train + num_val]
        test_pcaps = pcap_names[num_train + num_val:]
        
        # 为当前label创建目录
        label_dest_dir = os.path.join(dest_path, label)
        os.makedirs(label_dest_dir, exist_ok=True)
        
        # 保存三个txt文件，每个pcap名称占一行
        with open(os.path.join(label_dest_dir, "train.txt"), "w", encoding="utf-8") as f:
            for pcap_name in train_pcaps:
                f.write(pcap_name + "\n")
        
        with open(os.path.join(label_dest_dir, "val.txt"), "w", encoding="utf-8") as f:
            for pcap_name in val_pcaps:
                f.write(pcap_name + "\n")
        
        with open(os.path.join(label_dest_dir, "test.txt"), "w", encoding="utf-8") as f:
            for pcap_name in test_pcaps:
                f.write(pcap_name + "\n")
        
        print(f"已生成{label}类别的catalog，共{total}个文件（训练集:{len(train_pcaps)}, 验证集:{len(val_pcaps)}, 测试集:{len(test_pcaps)}）")
        
        # 清理内存
        del pcap_names, train_pcaps, val_pcaps, test_pcaps
        gc.collect()

    print(f"每个标签采样{str(k)}个流(txt文件)，已保存到: {dest_path}")

def generate_finetuning_dataset(preprocess_path: str, catalog_path: str = "", dest_path: str = "", packet_num_in_flow: int = 5):
    """
    生成微调数据集。支持两种模式：
    1. catalog模式（catalog_path非空）：从catalog_path读取每个label的train.txt, val.txt, test.txt，
       根据其中的pcap文件名从preprocess_path中读取对应的txt文件并生成微调数据集。
    2. 目录模式（catalog_path为空）：preprocess_path下已存在train/val/test的目录划分，
       每个划分目录下有label子目录，直接读取所有txt文件生成数据集。缺失的划分会跳过。

    Args:
        preprocess_path (str): 预处理文件的根目录
        catalog_path (str): catalog文件所在目录，为空则使用目录模式
        dest_path (str): 保存微调数据集的目的地目录
        packet_num_in_flow (int): 每个流包含的包数量
    """
    import os
    from .utils import _dump_in_chunks, _LM_input, _str_to_ids
    from tqdm import tqdm
    import sys
    import gc

    print("=" * 60)
    print("开始生成微调数据集")
    print("=" * 60)

    use_catalog = catalog_path and catalog_path.strip()
    split_names = ["train", "val", "test"]

    print(f"📂 预处理路径: {preprocess_path}")
    print(f"📂 目标路径: {dest_path}")
    print(f"📊 每流包数: {packet_num_in_flow}")
    print(f"🔧 模式: {'catalog模式' if use_catalog else '目录模式'}")
    if use_catalog:
        print(f"📂 Catalog路径: {catalog_path}")

    # 确保目标目录存在
    for split in split_names:
        os.makedirs(os.path.join(dest_path, split), exist_ok=True)
    print(f"✅ 已创建目标目录: {dest_path}/[train|val|test]")

    # 获取所有label名称
    if use_catalog:
        label_names = [name for name in os.listdir(catalog_path)
                       if os.path.isdir(os.path.join(catalog_path, name))]
        print(f"📋 从catalog检测到 {len(label_names)} 个标签: {label_names}")
    else:
        # 从 preprocess_path 下已存在的 train/val/test 目录中收集所有label
        label_set = set()
        for split in split_names:
            split_dir = os.path.join(preprocess_path, split)
            if os.path.isdir(split_dir):
                for name in os.listdir(split_dir):
                    if os.path.isdir(os.path.join(split_dir, name)):
                        label_set.add(name)
        label_names = sorted(label_set)
        detected_splits = [s for s in split_names if os.path.isdir(os.path.join(preprocess_path, s))]
        print(f"📋 目录模式: 从 {preprocess_path} 检测到划分 {detected_splits}")
        print(f"📋 检测到 {len(label_names)} 个标签: {label_names}")

    # 准备prompt
    print("🔧 准备prompt模板...")
    system_prompt = """<|im_start|>system
你是一个AI助手，擅长阅读表格形式的网络流量并对其进行思考和理解，并能够完成各种针对网络流量的问题。<|im_end|> """
    prompt = system_prompt + f"""
<|im_start|>user
接下来会给出一个流量表格，包含若干个包的头部特征和统计特征，以及在最后一列的payload。请输出对应的类别。
类别包含: {", ".join(label_names)}。
接下来是表格：<表格开始>"""
    prompt_ids = _str_to_ids(prompt, type="qwen3vl")[0]
    prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
给定流的类别是："""
    prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
    print("✅ Prompt模板准备完成")

    total_samples = {"train": 0, "val": 0, "test": 0}

    def process_txt_files(txt_filepaths, label, label_ids, dataset_name):
        """处理一组txt文件，生成样本并保存"""
        samples = []
        for txt_filepath in tqdm(txt_filepaths, desc=f"  处理 {label}/{dataset_name}", file=sys.stdout):
            try:
                lines = open(txt_filepath, "r", encoding="utf-8").readlines()
                assert len(lines) >= 3, f"文件行数小于3: {txt_filepath}"
                lines_used_here = packet_num_in_flow
                sample = _LM_input(lines[:lines_used_here], None, None, label_ids, prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True)
                while sample["data"][-1].shape[1] > 4096 and lines_used_here > 0:
                    lines_used_here -= 1
                    sample = _LM_input(lines[:lines_used_here], None, None, label_ids, prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True)
                assert sample["data"][-1].shape[1] <= 4096, f"样本长度大于4096: {txt_filepath}"
                samples.append(sample)
            except Exception as e:
                print(f"    ⚠️ 处理文件失败 {txt_filepath}: {e}")
                continue
        if samples:
            _dump_in_chunks(samples, os.path.join(dest_path, dataset_name), -1, name=f"{dataset_name}_{label}")
        return len(samples)

    print("\n" + "=" * 60)
    print("开始处理各标签数据")
    print("=" * 60)

    for idx, label in enumerate(label_names, 1):
        print(f"\n[{idx}/{len(label_names)}] 🏷️  处理标签: {label}")
        label_ids = _str_to_ids(label+"<|im_end|>", type="qwen3vl")[0]

        if use_catalog:
            # catalog模式：从catalog读取pcap文件名，在preprocess_path/label/下找对应txt
            label_dir = os.path.join(preprocess_path, label)
            catalog_label_dir = os.path.join(catalog_path, label)
            assert os.path.exists(catalog_label_dir) and os.path.isdir(catalog_label_dir), f"catalog目录不存在: {catalog_label_dir}"
            assert os.path.exists(label_dir) and os.path.isdir(label_dir), f"label目录不存在: {label_dir}"

            for catalog_filename, dataset_name in [("train.txt", "train"), ("val.txt", "val"), ("test.txt", "test")]:
                catalog_file = os.path.join(catalog_label_dir, catalog_filename)
                with open(catalog_file, "r", encoding="utf-8") as f:
                    pcap_names = [line.strip() for line in f if line.strip()]
                txt_filepaths = []
                for pcap_name in pcap_names:
                    txt_filename = pcap_name.rsplit('.', 1)[0] + ".txt"
                    txt_filepath = os.path.join(label_dir, txt_filename)
                    assert os.path.exists(txt_filepath), f"文件不存在: {txt_filepath}"
                    txt_filepaths.append(txt_filepath)
                n = process_txt_files(txt_filepaths, label, label_ids, dataset_name)
                assert n == len(pcap_names), f"样本数量不匹配: {catalog_file} {n} != {len(pcap_names)}"
                gc.collect()
        else:
            # 目录模式：直接从 preprocess_path/split/label/ 下读取所有txt
            for dataset_name in split_names:
                split_label_dir = os.path.join(preprocess_path, dataset_name, label)
                if not os.path.isdir(split_label_dir):
                    continue
                txt_filepaths = sorted([
                    os.path.join(split_label_dir, f)
                    for f in os.listdir(split_label_dir) if f.endswith(".txt")
                ])
                if not txt_filepaths:
                    continue
                print(f"  📄 {dataset_name}: 发现 {len(txt_filepaths)} 个txt文件")
                n = process_txt_files(txt_filepaths, label, label_ids, dataset_name)
                total_samples[dataset_name] += n
                print(f"    ✅ 生成 {n} 个样本")
                gc.collect()

    print("\n" + "=" * 60)
    print("微调数据集生成完成!")
    print("=" * 60)
    print(f"📊 样本统计:")
    print(f"   训练集: {total_samples['train']} 个样本")
    print(f"   验证集: {total_samples['val']} 个样本")
    print(f"   测试集: {total_samples['test']} 个样本")
    print(f"   总计: {sum(total_samples.values())} 个样本")
    print(f"📂 数据保存至: {dest_path}")

if __name__ == '__main__':
    from fire import Fire
    Fire()