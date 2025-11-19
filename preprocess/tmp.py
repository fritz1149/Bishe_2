from .utils import _cut_bursts
import sys
import os
from tqdm import tqdm
import sqlite3

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

class IdGenerator:
    """
    进程安全的 ID 生成器
    只返回 ID 号段 (start_id, size)，避免传递大量数据
    
    进程安全性说明：
    - 使用 multiprocessing.Manager 创建的 Value 和 Lock 是代理对象（ValueProxy 和 AcquirerProxy）
    - 当 IdGenerator 对象被 pickle 序列化传递给子进程时：
      1. Manager 代理对象会被序列化（序列化的是连接信息，不是实际值）
      2. 子进程反序列化后，代理对象会重新连接到同一个 Manager 进程
      3. 所有子进程访问的是同一个共享的计数器和锁
    - 因此是进程安全的，不会创建副本
    - 注意：Manager 对象本身不能被序列化，所以不在 IdGenerator 中保存 manager 引用
       Manager 的生命周期由调用者管理（在 generate_classify_tmp 中保持引用直到最后）
    """
    def __init__(self, manager=None):
        # 使用 multiprocessing.Value 实现进程安全的计数器
        if manager is None:
            manager = multiprocessing.Manager()
        self._id = manager.Value('i', 0)
        self._lock = manager.Lock()
        # 注意：不保存 manager 引用，因为 Manager 对象本身不能被序列化
        # Manager 的生命周期由调用者管理（在 generate_classify_tmp 中保持引用）

    def get(self, size: int = 1):
        """
        获取 ID 号段
        
        Args:
            size: 需要的 ID 数量
            
        Returns:
            (start_id, size): ID 号段的起始值和大小
        """
        with self._lock:
            ret = self._id.value
            self._id.value += size
            return (ret, size)  # 只返回号段，不创建列表

# 将 process_batch_flows 移到模块顶层，使其可以被 pickle（多进程要求）
def process_batch_flows(args):
    """
    批量处理一批流文件，收集所有数据后一次性插入 SQLite 数据库
    
    Args:
        args: (batch_flows, flow_id_gen, burst_id_gen, packet_id_gen, payload_id_gen, 
               label_id_gen, label_name_to_id, db_path)
        - batch_flows: 一批流文件，list of (flow_path, label_name)
    """
    batch_flows, flow_id_gen, burst_id_gen, packet_id_gen, payload_id_gen, label_id_gen, label_name_to_id, tmp_path = args

    import sys
    import os
    import gc
    import sqlite3

    from .utils import _cut_bursts
    from .model import init_database

    pid = os.getpid()
    db_path = os.path.join(tmp_path, f"dataset_{pid}.db")
    # INSERT_YOUR_CODE
    import os
    if not os.path.exists(db_path):
        # 如果db_path对应的数据库文件不存在，则进行数据库初始化
        conn = init_database(db_path)
    else:
        # 每个进程独立连接数据库（SQLite 支持并发读，写会自动加锁）
        conn = sqlite3.connect(db_path)
    # conn.execute("PRAGMA foreign_keys = ON;")

    # 收集整批数据
    all_flow_rows = []
    all_burst_rows = []
    all_packet_rows = []
    all_payload_rows = []
    all_label_packet_rows = []
    all_label_packet_with_payload_rows = []
    all_label_flows_rows = []
    
    success_count = 0
    error_count = 0
    processed_flow_ids = []

    try:
        # 处理每个流文件，收集数据
        for flow_path, label_name in batch_flows:
            try:
                flow_type = None
                if "TCP" in flow_path:
                    flow_type = "TCP"
                elif "UDP" in flow_path:
                    flow_type = "UDP"
                else:
                    raise Exception(f"Unknown protocol type in flow file path: {flow_path}")
                # 读取文件数据
                bursts = _cut_bursts(flow_path)
                if not bursts:
                    error_count += 1
                    continue

                # 动态获取所需的 ID 号段（本次实际需要多少就申请多少）
                total_bursts = len(bursts)
                total_packets = sum(len(b["packets"]) for b in bursts)
                total_payloads = sum(len(b["payloads"]) for b in bursts)

                flow_id_start, flow_id_size = flow_id_gen.get(1)  # 一个 flow
                burst_id_start, burst_id_size = burst_id_gen.get(total_bursts)
                packet_id_start, packet_id_size = packet_id_gen.get(total_packets)
                payload_id_start, payload_id_size = payload_id_gen.get(total_payloads)

                # 验证 ID 号段是否足够
                assert burst_id_size >= total_bursts, f"Burst ID 号段不足: 需要 {total_bursts}, 分配 {burst_id_size}"
                assert packet_id_size >= total_packets, f"Packet ID 号段不足: 需要 {total_packets}, 分配 {packet_id_size}"
                assert payload_id_size >= total_payloads, f"Payload ID 号段不足: 需要 {total_payloads}, 分配 {payload_id_size}"

                flow_id = flow_id_start  # 只使用第一个 flow_id
                label_id = label_name_to_id[label_name]
                assert label_id is not None, f"Label '{label_name}' 的 ID 未找到"

                burst_rows = []
                packet_rows = []
                payload_rows = []
                label_packet_rows = []
                label_packet_with_payload_rows = []
                
                packet_id_current = packet_id_start
                payload_id_current = payload_id_start
                payload_ge1_burst_num = 0

                for burst_idx, b in enumerate(bursts):
                    burst_id = burst_id_start + burst_idx
                    payload_num = len(b["payloads"])
                    burst_rows.append((burst_id, flow_id, burst_idx, payload_num, label_name, flow_type))
                    if payload_num >= 1:
                        payload_ge1_burst_num += 1
                    payload_idx_in_burst = 0
                    for pkt_idx, pkt_line in enumerate(b["packets"]):
                        packet_id = packet_id_current + pkt_idx
                        payload_id = None
                        if payload_idx_in_burst < len(b["payloads_index"]) and pkt_idx == b["payloads_index"][payload_idx_in_burst]:
                            payload_id = payload_id_current + payload_idx_in_burst
                            payload_content = b["payloads"][payload_idx_in_burst]
                            payload_rows.append((payload_id, packet_id, label_id, payload_content, flow_type))
                            label_packet_with_payload_rows.append((label_id, packet_id))
                            payload_idx_in_burst += 1
                        packet_rows.append((packet_id, burst_id, pkt_idx, payload_id, pkt_line, label_name, flow_type))
                        label_packet_rows.append((label_id, packet_id))
                    packet_id_current += len(b["packets"])
                    payload_id_current += len(b["payloads"])

                # 添加到批量插入列表
                all_flow_rows.append((flow_id, label_name, payload_ge1_burst_num, flow_type, flow_path))
                all_burst_rows.extend(burst_rows)
                all_packet_rows.extend(packet_rows)
                all_payload_rows.extend(payload_rows)
                all_label_packet_rows.extend(label_packet_rows)
                all_label_packet_with_payload_rows.extend(label_packet_with_payload_rows)
                all_label_flows_rows.append((label_id, flow_id))
                
                processed_flow_ids.append(flow_id)
                success_count += 1
                
                # 立即释放内存
                del bursts, burst_rows, packet_rows, payload_rows
                del label_packet_rows, label_packet_with_payload_rows
                
            except Exception as e:
                error_count += 1
                import traceback
                print(f"\n[❌ Exception] Error occurred while processing file:\n  Path  : {flow_path}\n  Label : {label_name}\n  Error : {repr(e)}")
                print("详细异常信息如下：")
                traceback.print_exc()
        
        # 使用插入锁确保同一时间只有一个进程在执行数据库写入操作
        # 一次性批量插入整批数据
        if all_flow_rows:  # 只有成功处理了至少一个流文件才插入
            conn.execute("BEGIN TRANSACTION;")
            try:
                # 批量插入 flows
                conn.executemany(
                    "INSERT INTO flows (id, label, payload_ge1_burst_num, flow_type, flow_path) VALUES (?, ?, ?, ?, ?)",
                    all_flow_rows
                )

                # 批量插入其他表
                if all_burst_rows:
                    conn.executemany(
                        "INSERT INTO bursts (id, flow_id, index_in_flow, payload_num, label, flow_type) VALUES (?, ?, ?, ?, ?, ?)",
                        all_burst_rows
                    )
                if all_packet_rows:
                    conn.executemany(
                        "INSERT INTO packets (id, burst_id, index_in_burst, payload_id, line, label, flow_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        all_packet_rows
                    )
                if all_payload_rows:
                    conn.executemany(
                        "INSERT INTO payloads (id, packet_id, label, content, flow_type) VALUES (?, ?, ?, ?, ?)",
                        all_payload_rows
                    )
                if all_label_packet_rows:
                    conn.executemany(
                        "INSERT OR IGNORE INTO label_packets (label_id, packet_id) VALUES (?, ?)",
                        all_label_packet_rows
                    )
                if all_label_packet_with_payload_rows:
                    conn.executemany(
                        "INSERT OR IGNORE INTO label_packets_with_payload (label_id, packet_id) VALUES (?, ?)",
                        all_label_packet_with_payload_rows
                    )
                if all_label_flows_rows:
                    conn.executemany(
                        "INSERT OR IGNORE INTO label_flows (label_id, flow_id) VALUES (?, ?)",
                        all_label_flows_rows
                    )
                conn.commit()
                # print(f"批量插入 {len(processed_flow_ids)} 个流成功: {processed_flow_ids}")
                print(f"{pid}进程批量插入成功")
                sys.stdout.flush()
            except Exception as e:
                conn.rollback()
                raise e

        # 释放内存
        del all_flow_rows, all_burst_rows, all_packet_rows, all_payload_rows
        del all_label_packet_rows, all_label_packet_with_payload_rows, all_label_flows_rows
        gc.collect()
        conn.close()
        
        return {
            "success": True,
            "success_count": success_count,
            "error_count": error_count,
            "flow_ids": processed_flow_ids
        }
    except Exception as e:
        conn.rollback()
        conn.close()
        import traceback
        print(f"\n[❌ Exception] Error occurred while processing batch:\n  Batch size: {len(batch_flows)}\n  Error : {repr(e)}")
        print("详细异常信息如下：")
        traceback.print_exc()
        return {
            "success": False,
            "success_count": success_count,
            "error_count": error_count,
            "flow_ids": []
        }
def generate_classify_tmp(src_path: str, tmp_path: str, max_workers: int = None, workers: int = 1, batch_size: int = 50):   
    """
    生成分类数据集中间文件（使用 SQLite 数据库）
    
    关键优化：
    1. 使用 SQLite 数据库代替 pickle 文件
    2. 进程只获取 ID 号段，不传递大量数据
    3. 每个进程独立连接数据库，插入后立即释放内存
    4. 使用事务批量插入，提高性能
    """
    import gc
    
    if workers > 0:
        # INSERT_YOUR_CODE
        # 若tmp_path存在文件，则全部删除
        if os.path.exists(tmp_path):
            for fname in os.listdir(tmp_path):
                fpath = os.path.join(tmp_path, fname)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        print(f"无法删除文件 {fpath}: {e}")
        # 创建数据库路径
        os.makedirs(tmp_path, exist_ok=True)
        
        # 创建进程管理器，用于共享ID生成器和插入锁
        manager = multiprocessing.Manager()
        
        # 针对5类对象的IdGenerator（使用Manager实现进程安全）
        flow_id_gen = IdGenerator(manager)
        burst_id_gen = IdGenerator(manager)
        packet_id_gen = IdGenerator(manager)
        payload_id_gen = IdGenerator(manager)
        label_id_gen = IdGenerator(manager)

        # 收集所有流文件并预先插入 labels
        flow_files = []    
        labels = []
        label_name_to_id = {}
        
        # 先连接数据库，插入所有 labels
        for label_name in os.listdir(src_path):
            label_dir = os.path.join(src_path, label_name)
            if os.path.isdir(label_dir):
                labels.append(label_name)
                for fname in os.listdir(label_dir):
                    if fname.lower().endswith('.txt'):
                        flow_path = os.path.join(label_dir, fname)
                        if os.path.isfile(flow_path):
                            flow_files.append((flow_path, label_name))
        
        # INSERT_YOUR_CODE
        # 为每个 label_name 分配一个唯一的 label_id，并维护 label_name_to_id 字典
        for idx, label_name in enumerate(labels):
            label_name_to_id[label_name] = idx
    
        print(f"已收集 {len(flow_files)} 个流文件，{len(labels)} 个标签")
        
        # 将流文件分批
        batches = []
        for i in range(0, len(flow_files), batch_size):
            batches.append(flow_files[i:i+batch_size])
        
        print(f"将流文件分为 {len(batches)} 批，每批约 {batch_size} 个文件")
        
        # 并发处理流文件批次
        # 关键优化：进程批量处理一批流文件，收集数据后一次性插入数据库
        total_files = len(flow_files)
        main_pbar = tqdm(total=total_files, desc="处理Flow文件", position=0, file=sys.stdout, leave=True, mininterval=0.1, dynamic_ncols=True)
        
        # 统计信息
        success_count = 0
        error_count = 0
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # 为每个批次提交任务
            futures = []
            for batch in batches:
                # 传递批次数据和id生成器对象、插入锁
                args = (batch, flow_id_gen, burst_id_gen, packet_id_gen, payload_id_gen, label_id_gen, label_name_to_id, tmp_path)
                futures.append(executor.submit(process_batch_flows, args))
            
            # 等待所有任务完成
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    if result.get("success"):
                        batch_success = result.get("success_count", 0)
                        batch_error = result.get("error_count", 0)
                        success_count += batch_success
                        error_count += batch_error
                        main_pbar.update(batch_success + batch_error)
                    else:
                        batch_error = result.get("error_count", 0)
                        error_count += batch_error
                        main_pbar.update(batch_error)
                except Exception as e:
                    error_count += batch_size  # 粗略估算，实际可能有偏差
                    import traceback
                    print(f"\n[❌ Exception] Error occurred while processing future result:\n  Error : {repr(e)}")
                    print("详细异常信息如下：")
                    traceback.print_exc()
                    main_pbar.update(batch_size)  # 粗略估算
            
            # 释放future对象
            del futures
            gc.collect()
            
            main_pbar.close()

        # 关闭Manager，释放资源
        manager.shutdown()
        del manager, flow_id_gen, burst_id_gen, packet_id_gen, payload_id_gen, label_id_gen
        gc.collect()
    
    else:
        labels = []
        label_name_to_id = {}
        for label_name in os.listdir(src_path):
            label_dir = os.path.join(src_path, label_name)
            if os.path.isdir(label_dir):
                labels.append(label_name)
        for idx, label_name in enumerate(labels):
            label_name_to_id[label_name] = idx
    
    # INSERT_YOUR_CODE
    from .model import execute_sql_on_dbs, connect_to_dbs, close_dbs

    # 从数据库统计结果（不需要将所有数据加载到内存）
    dbs = connect_to_dbs(tmp_path)
    # INSERT_YOUR_CODE
    # 将labels信息插入labels表
    # label_rows = [(label_id, label_name) for label_name, label_id in label_name_to_id.items()]
    # INSERT_YOUR_CODE
    import json
    with open(f"{tmp_path}/label_name_to_id.json", "w", encoding="utf-8") as f:
        json.dump(label_name_to_id, f, ensure_ascii=False, indent=2)
    print("\n统计数据库结果...")
    
    # 统计各表数量
    flow_count = sum(execute_sql_on_dbs(dbs, "SELECT COUNT(*) FROM flows", unpack=True))
    burst_count = sum(execute_sql_on_dbs(dbs, "SELECT COUNT(*) FROM bursts", unpack=True))
    packet_count = sum(execute_sql_on_dbs(dbs, "SELECT COUNT(*) FROM packets", unpack=True))
    payload_count = sum(execute_sql_on_dbs(dbs, "SELECT COUNT(*) FROM payloads", unpack=True))
    label_count = len(labels)
    
    print(f"flows: {flow_count}")
    print(f"bursts: {burst_count}")
    print(f"packets: {packet_count}")
    print(f"payloads: {payload_count}")
    print(f"labels: {label_count}")
    
    # 统计每个 label 的信息
    print("\n各标签统计:")
    for label_name, label_id in label_name_to_id.items():
        packet_count = sum(execute_sql_on_dbs(dbs, f"SELECT COUNT(*) FROM label_packets WHERE label_id = {label_id}", unpack=True))
        packet_with_payload_count = sum(execute_sql_on_dbs(dbs, f"SELECT COUNT(*) FROM label_packets_with_payload WHERE label_id = {label_id}", unpack=True))
        flow_count = sum(execute_sql_on_dbs(dbs, f"SELECT COUNT(*) FROM label_flows WHERE label_id = {label_id}", unpack=True))
        print(f"  Label '{label_name}': {packet_count} packet_ids, {packet_with_payload_count} packet_with_payload_ids, {flow_count} flow_ids")
    
    close_dbs(dbs)
    
    if workers > 0:
        print(f"\n处理完成: 成功 {success_count}, 失败 {error_count}")

if __name__ == "__main__":
    from fire import Fire
    Fire()