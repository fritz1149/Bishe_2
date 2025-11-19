"""
SQLite 模型使用示例
演示如何使用重构后的 model.py 创建和使用数据库
"""

from .model import init_database, print_database_schema
import sqlite3

# 示例1: 初始化数据库
def example_init_database():
    """初始化数据库示例"""
    db_path = "example.db"
    conn = init_database(db_path)
    
    # 打印数据库结构
    print_database_schema(conn)
    
    # 关闭连接
    conn.close()
    print(f"\n数据库已创建: {db_path}")

# 示例2: 插入数据
def example_insert_data():
    """插入数据示例"""
    db_path = "example.db"
    conn = init_database(db_path)
    
    # 插入一个 flow
    conn.execute("INSERT INTO flows (id, label, payload_ge1_burst_num) VALUES (?, ?, ?)", 
                 (1, "example_label", 2))
    
    # 插入一个 burst
    conn.execute("INSERT INTO bursts (id, flow_id, index_in_flow, payload_num, label) VALUES (?, ?, ?, ?, ?)",
                 (1, 1, 0, 3, "example_label"))
    
    # 插入一个 packet
    conn.execute("INSERT INTO packets (id, burst_id, index_in_burst, line, label) VALUES (?, ?, ?, ?, ?)",
                 (1, 1, 0, "example packet line", "example_label"))
    
    # 插入一个 payload
    conn.execute("INSERT INTO payloads (id, packet_id, label, content) VALUES (?, ?, ?, ?)",
                 (1, 1, 0, "example payload content"))
    
    # 更新 packet 的 payload_id
    conn.execute("UPDATE packets SET payload_id = ? WHERE id = ?", (1, 1))
    
    # 插入关联关系
    conn.execute("INSERT INTO flow_bursts (flow_id, burst_id) VALUES (?, ?)", (1, 1))
    conn.execute("INSERT INTO burst_packets (burst_id, packet_id) VALUES (?, ?)", (1, 1))
    conn.execute("INSERT INTO burst_payloads (burst_id, payload_id) VALUES (?, ?)", (1, 1))
    
    conn.commit()
    conn.close()
    print("数据插入完成")

# 示例3: 查询数据
def example_query_data():
    """查询数据示例"""
    db_path = "example.db"
    conn = init_database(db_path)
    
    # 查询所有 flows
    flows = conn.execute("SELECT * FROM flows").fetchall()
    print(f"Flows: {flows}")
    
    # 查询某个 flow 的所有 bursts
    bursts = conn.execute("""
        SELECT b.* FROM bursts b
        WHERE b.flow_id = ?
        ORDER BY b.index_in_flow
    """, (1,)).fetchall()
    print(f"Bursts for flow 1: {bursts}")
    
    # 查询某个 burst 的所有 packets
    packets = conn.execute("""
        SELECT p.* FROM packets p
        WHERE p.burst_id = ?
        ORDER BY p.index_in_burst
    """, (1,)).fetchall()
    print(f"Packets for burst 1: {packets}")
    
    # 查询某个 packet 的 payload
    payload = conn.execute("""
        SELECT pl.* FROM payloads pl
        WHERE pl.packet_id = ?
    """, (1,)).fetchone()
    print(f"Payload for packet 1: {payload}")
    
    conn.close()

if __name__ == "__main__":
    print("=" * 80)
    print("SQLite 模型使用示例")
    print("=" * 80)
    
    print("\n1. 初始化数据库")
    example_init_database()
    
    print("\n2. 插入数据")
    example_insert_data()
    
    print("\n3. 查询数据")
    example_query_data()
