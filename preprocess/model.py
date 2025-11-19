"""
SQLite 数据库表结构定义
将原有的 Python 对象模型转换为 SQLite 表结构
"""

import sqlite3
from typing import Optional

# ============================================================================
# SQL 表创建语句
# ============================================================================

# 1. flows 表：存储流信息
CREATE_FLOWS_TABLE = """
CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    payload_ge1_burst_num INTEGER DEFAULT 0,
    flow_type TEXT NOT NULL,
    flow_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# 2. bursts 表：存储 burst 信息
CREATE_BURSTS_TABLE = """
CREATE TABLE IF NOT EXISTS bursts (
    id INTEGER PRIMARY KEY,
    flow_id INTEGER NOT NULL,
    index_in_flow INTEGER NOT NULL,
    payload_num INTEGER DEFAULT 0,
    label TEXT NOT NULL,
    flow_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (flow_id) REFERENCES flows(id) ON DELETE CASCADE,
    UNIQUE(flow_id, index_in_flow)
);
"""

# 3. packets 表：存储 packet 信息
# 注意：payload_id 外键约束将在创建 payloads 表后添加
CREATE_PACKETS_TABLE = """
CREATE TABLE IF NOT EXISTS packets (
    id INTEGER PRIMARY KEY,
    burst_id INTEGER NOT NULL,
    index_in_burst INTEGER NOT NULL,
    payload_id INTEGER,
    line TEXT NOT NULL,
    label TEXT NOT NULL,
    flow_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (burst_id) REFERENCES bursts(id) ON DELETE CASCADE,
    UNIQUE(burst_id, index_in_burst)
);
"""

# 4. payloads 表：存储 payload 内容
CREATE_PAYLOADS_TABLE = """
CREATE TABLE IF NOT EXISTS payloads (
    id INTEGER PRIMARY KEY,
    packet_id INTEGER NOT NULL,
    label INTEGER NOT NULL,
    content TEXT NOT NULL,
    flow_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (packet_id) REFERENCES packets(id) ON DELETE CASCADE,
    UNIQUE(packet_id)
);
"""

# 添加 packets.payload_id 的外键约束（在 payloads 表创建后）
# ADD_PACKETS_PAYLOAD_FK = """
# -- 注意：SQLite 不支持 ALTER TABLE ADD FOREIGN KEY，需要在创建表时定义
# -- 或者使用触发器来实现约束
# """

# 5. labels 表：存储标签元信息
CREATE_LABELS_TABLE = """
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# 6. flow_bursts 关联表：flows 和 bursts 的多对多关系（实际是一对多，但便于查询）
# 注意：bursts 表中已有 flow_id，此表可选，用于快速查询某个 flow 的所有 bursts
# CREATE_FLOW_BURSTS_TABLE = """
# CREATE TABLE IF NOT EXISTS flow_bursts (
#     flow_id INTEGER NOT NULL,
#     burst_id INTEGER NOT NULL,
#     PRIMARY KEY (flow_id, burst_id),
#     FOREIGN KEY (flow_id) REFERENCES flows(id) ON DELETE CASCADE,
#     FOREIGN KEY (burst_id) REFERENCES bursts(id) ON DELETE CASCADE
# );
# """

# 7. burst_packets 关联表：bursts 和 packets 的关系（实际是一对多）
# CREATE_BURST_PACKETS_TABLE = """
# CREATE TABLE IF NOT EXISTS burst_packets (
#     burst_id INTEGER NOT NULL,
#     packet_id INTEGER NOT NULL,
#     PRIMARY KEY (burst_id, packet_id),
#     FOREIGN KEY (burst_id) REFERENCES bursts(id) ON DELETE CASCADE,
#     FOREIGN KEY (packet_id) REFERENCES packets(id) ON DELETE CASCADE
# );
# """

# 8. burst_payloads 关系视图：基于 packets、payloads，自动关联 burst 和 payload
CREATE_BURST_PAYLOADS_VIEW = """
CREATE VIEW IF NOT EXISTS burst_payloads AS
SELECT
    p.burst_id AS burst_id,
    pl.id AS payload_id
FROM
    packets p
INNER JOIN
    payloads pl
ON
    p.id = pl.packet_id
;
"""

# INSERT_YOUR_CODE
# 9. burst_packet_count 视图：每个 burst 包含的 packet 数量
CREATE_BURST_PACKET_COUNT_VIEW = """
CREATE VIEW IF NOT EXISTS burst_packet_count AS
SELECT
    burst_id,
    COUNT(*) AS packet_count
FROM
    packets
GROUP BY
    burst_id
;
"""


# 9. label_packets 关联表：labels 和 packets 的多对多关系
CREATE_LABEL_PACKETS_TABLE = """
CREATE TABLE IF NOT EXISTS label_packets (
    label_id INTEGER NOT NULL,
    packet_id INTEGER NOT NULL,
    PRIMARY KEY (label_id, packet_id),
    FOREIGN KEY (label_id) REFERENCES labels(id) ON DELETE CASCADE,
    FOREIGN KEY (packet_id) REFERENCES packets(id) ON DELETE CASCADE
);
"""

# 10. label_packets_with_payload 关联表：labels 和带 payload 的 packets
CREATE_LABEL_PACKETS_WITH_PAYLOAD_TABLE = """
CREATE TABLE IF NOT EXISTS label_packets_with_payload (
    label_id INTEGER NOT NULL,
    packet_id INTEGER NOT NULL,
    PRIMARY KEY (label_id, packet_id),
    FOREIGN KEY (label_id) REFERENCES labels(id) ON DELETE CASCADE,
    FOREIGN KEY (packet_id) REFERENCES packets(id) ON DELETE CASCADE
);
"""

# 11. label_flows 关联表：labels 和 flows 的多对多关系
CREATE_LABEL_FLOWS_TABLE = """
CREATE TABLE IF NOT EXISTS label_flows (
    label_id INTEGER NOT NULL,
    flow_id INTEGER NOT NULL,
    PRIMARY KEY (label_id, flow_id),
    FOREIGN KEY (label_id) REFERENCES labels(id) ON DELETE CASCADE,
    FOREIGN KEY (flow_id) REFERENCES flows(id) ON DELETE CASCADE
);
"""

# ============================================================================
# 索引创建语句（提高查询性能）
# ============================================================================

CREATE_INDEXES = [
    # flows 表索引
    # "CREATE INDEX IF NOT EXISTS idx_flows_label ON flows(label);",
    
    # bursts 表索引
    "CREATE INDEX IF NOT EXISTS idx_bursts_flow_id ON bursts(flow_id);",
    # "CREATE INDEX IF NOT EXISTS idx_bursts_label ON bursts(label);",
    "CREATE INDEX IF NOT EXISTS idx_bursts_payload_num ON bursts(payload_num);",
    
    # packets 表索引
    "CREATE INDEX IF NOT EXISTS idx_packets_burst_id ON packets(burst_id);",
    "CREATE INDEX IF NOT EXISTS idx_packets_payload_id ON packets(payload_id);",
    # "CREATE INDEX IF NOT EXISTS idx_packets_label ON packets(label);",
    
    # payloads 表索引
    "CREATE INDEX IF NOT EXISTS idx_payloads_packet_id ON payloads(packet_id);",
    # "CREATE INDEX IF NOT EXISTS idx_payloads_label ON payloads(label);",
    
    # 关联表索引（注意：flow_bursts 和 burst_packets 表已被注释，使用外键索引代替）
    # burst_payloads 是视图，无法创建索引，但可以通过 packets.burst_id 和 payloads.packet_id 索引查询
    "CREATE INDEX IF NOT EXISTS idx_label_packets_label_id ON label_packets(label_id);",
    "CREATE INDEX IF NOT EXISTS idx_label_packets_packet_id ON label_packets(packet_id);",
    "CREATE INDEX IF NOT EXISTS idx_label_packets_with_payload_label_id ON label_packets_with_payload(label_id);",
    "CREATE INDEX IF NOT EXISTS idx_label_packets_with_payload_packet_id ON label_packets_with_payload(packet_id);",
    "CREATE INDEX IF NOT EXISTS idx_label_flows_label_id ON label_flows(label_id);",
    "CREATE INDEX IF NOT EXISTS idx_label_flows_flow_id ON label_flows(flow_id);",
]

# 对应 CREATE_INDEXES 的 drop 索引语句（便于数据库重置时清理索引）
DROP_INDEXES = [
    "DROP INDEX IF EXISTS idx_bursts_flow_id;",
    "DROP INDEX IF EXISTS idx_bursts_payload_num;",
    "DROP INDEX IF EXISTS idx_packets_burst_id;",
    "DROP INDEX IF EXISTS idx_packets_payload_id;",
    "DROP INDEX IF EXISTS idx_payloads_packet_id;",
    "DROP INDEX IF EXISTS idx_label_packets_label_id;",
    "DROP INDEX IF EXISTS idx_label_packets_packet_id;",
    "DROP INDEX IF EXISTS idx_label_packets_with_payload_label_id;",
    "DROP INDEX IF EXISTS idx_label_packets_with_payload_packet_id;",
    "DROP INDEX IF EXISTS idx_label_flows_label_id;",
    "DROP INDEX IF EXISTS idx_label_flows_flow_id;",
]


# ============================================================================
# 数据库初始化函数
# ============================================================================

def init_database(db_path: str) -> sqlite3.Connection:
    """
    初始化 SQLite 数据库，创建所有表和索引
    
    Args:
        db_path: 数据库文件路径
        
    Returns:
        sqlite3.Connection: 数据库连接对象
    """
    # 若db_path已经存在，则清空其中内容，否则创建对应的目录
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF;")  # 启用外键约束
    
    # 创建所有表（注意顺序：先创建被引用的表）
    # 1. 基础表
    conn.execute(CREATE_FLOWS_TABLE)
    conn.execute(CREATE_BURSTS_TABLE)
    conn.execute(CREATE_PACKETS_TABLE)  # payload_id 暂时不设外键约束
    conn.execute(CREATE_PAYLOADS_TABLE)
    conn.execute(CREATE_LABELS_TABLE)
    
    # 2. 关联表
    # conn.execute(CREATE_FLOW_BURSTS_TABLE)
    # conn.execute(CREATE_BURST_PACKETS_TABLE)
    # conn.execute(CREATE_BURST_PAYLOADS_TABLE)
    conn.execute(CREATE_BURST_PAYLOADS_VIEW)
    conn.execute(CREATE_BURST_PACKET_COUNT_VIEW)
    conn.execute(CREATE_LABEL_PACKETS_TABLE)
    conn.execute(CREATE_LABEL_PACKETS_WITH_PAYLOAD_TABLE)
    conn.execute(CREATE_LABEL_FLOWS_TABLE)
    
    # 注意：SQLite 不支持 ALTER TABLE ADD FOREIGN KEY
    # packets.payload_id 的外键约束已在 CREATE_PACKETS_TABLE 中移除
    # 可以通过应用层逻辑或触发器来保证数据完整性
    
    # 创建所有索引
    for index_sql in CREATE_INDEXES:
        conn.execute(index_sql)
    
    conn.commit()
    return conn

# ============================================================================
# 辅助函数：获取表结构信息
# ============================================================================

def get_table_schema(conn: sqlite3.Connection, table_name: str) -> list:
    """
    获取表的架构信息
    
    Args:
        conn: 数据库连接
        table_name: 表名
        
    Returns:
        表的列信息列表
    """
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

def print_database_schema(conn: sqlite3.Connection):
    """
    打印数据库所有表的结构信息
    
    Args:
        conn: 数据库连接
    """
    tables = [
        'flows', 'bursts', 'packets', 'payloads', 'labels',
        'label_packets', 'label_packets_with_payload', 'label_flows'
    ]
    
    for table in tables:
        print(f"\n表: {table}")
        print("-" * 80)
        schema = get_table_schema(conn, table)
        for col in schema:
            print(f"  {col[1]:<20} {col[2]:<15} {'NOT NULL' if col[3] else 'NULL':<10} {'PK' if col[5] else '':<5}")

def connect_to_dbs(db_path: str) -> list[sqlite3.Connection]:
    """
    连接到多个数据库
    
    Args:
        db_paths: 数据库文件路径列表
        
    Returns:
        list[sqlite3.Connection]: 数据库连接对象列表
    """
    import os
    db_files = []
    if os.path.isdir(db_path):
        # 获取db_path目录下所有以.db结尾的文件，绝对路径
        db_files = [
            os.path.join(db_path, f)
            for f in os.listdir(db_path)
            if os.path.isfile(os.path.join(db_path, f)) and f.endswith('.db')
        ]
    elif os.path.isfile(db_path):
        db_files = [db_path]
    else:
        raise ValueError(f"db_path {db_path} 不是一个有效的文件或目录！")
    conns = [sqlite3.connect(p, check_same_thread=False) for p in db_files]
    # for conn in conns:
    #     conn.execute('PRAGMA synchronous = NORMAL;')
        # cursor = conn.cursor()
        # cursor.execute('PRAGMA integrity_check;')
        # result = cursor.fetchone()
        # print(result)
        # conn.execute("PRAGMA check_same_thread = FALSE;")
        # conn.execute("PRAGMA schema.mode = 'rw';")
    return conns

def close_dbs(conns: list[sqlite3.Connection]):
    """
    关闭多个数据库连接
    
    Args:
        dbs: 数据库连接列表
    """
    for conn in conns:
        conn.close()

import threading

def execute_sql_on_dbs(conns: list[sqlite3.Connection], sql: str, unpack: bool = False, parallel: bool = True):
    """
    在多个数据库上执行 SQL 语句，可并行/串行

    Args:
        conns: 数据库连接列表
        sql: SQL 语句
        unpack: 是否解包，即返回结果的第一个元素
        parallel: 是否并行执行
    """
    results = []
    if parallel:
        results_lock = threading.Lock()

        def worker(conn):
            try:
                cursor = conn.execute(sql)
                rows = cursor.fetchall()
                with results_lock:
                    results.extend(rows)
            except Exception as e:
                print(f"[SQL线程异常] 执行SQL时出错: {e}\nSQL: {sql}")

        threads = []
        for conn in conns:
            t = threading.Thread(target=worker, args=(conn,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    else:
        for conn in conns:
            try:
                cursor = conn.execute(sql)
                rows = cursor.fetchall()
                results.extend(rows)
            except Exception as e:
                print(f"[SQL异常] 执行SQL时出错: {e}\nSQL: {sql}")
    if unpack:
        results = [row[0] for row in results]
    return results
def add_view(src_path: str):
    import os

    # 遍历src_path下的一级目录
    if not os.path.isdir(src_path):
        raise ValueError(f"{src_path} 不是一个有效的目录。")
    subdirs = [os.path.join(src_path, name) for name in os.listdir(src_path)
               if os.path.isdir(os.path.join(src_path, name))]
    for subdir in subdirs:
        conns = connect_to_dbs(subdir)
        for i,conn in enumerate(conns):
            print(f"创建视图 {i}...")
            conn.execute(CREATE_BURST_PACKET_COUNT_VIEW)
            conn.commit()
        close_dbs(conns)
    print("视图创建完成")

if __name__ == "__main__":
    import fire
    fire.Fire()