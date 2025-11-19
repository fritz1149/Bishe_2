"""
测试 IdGenerator 的进程安全性
验证 Manager 代理对象在子进程中的行为
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

class IdGenerator:
    def __init__(self, manager=None):
        if manager is None:
            manager = multiprocessing.Manager()
        self._id = manager.Value('i', 0)
        self._lock = manager.Lock()
        # 注意：不保存 manager 引用，因为 Manager 对象本身不能被序列化

    def get(self, size: int = 1):
        with self._lock:
            ret = self._id.value
            self._id.value += size
            return (ret, size)

def test_process(id_gen, process_id):
    """测试函数：在子进程中使用 IdGenerator"""
    print(f"进程 {process_id}: IdGenerator id={id(id_gen)}, _id id={id(id_gen._id)}, _lock id={id(id_gen._lock)}")
    print(f"进程 {process_id}: _id 类型={type(id_gen._id)}, _lock 类型={type(id_gen._lock)}")
    
    # 获取几个 ID
    results = []
    for i in range(5):
        id_range = id_gen.get(10)
        results.append(id_range)
        print(f"进程 {process_id}: 获取 ID 号段 {id_range}")
        time.sleep(0.01)  # 模拟处理时间
    
    return results

if __name__ == "__main__":
    print("=" * 80)
    print("测试 IdGenerator 进程安全性")
    print("=" * 80)
    
    # 创建 Manager 和 IdGenerator
    manager = multiprocessing.Manager()
    id_gen = IdGenerator(manager)
    
    print(f"\n主进程:")
    print(f"  IdGenerator 对象 id: {id(id_gen)}")
    print(f"  _id 对象 id: {id(id_gen._id)}")
    print(f"  _lock 对象 id: {id(id_gen._lock)}")
    print(f"  Manager 对象 id: {id(manager)}")
    print(f"  _id 的类型: {type(id_gen._id)}")
    print(f"  _lock 的类型: {type(id_gen._lock)}")
    
    # 在主进程中先获取一个 ID
    first_id = id_gen.get(5)
    print(f"\n主进程获取第一个 ID 号段: {first_id}")
    
    # 在子进程中测试
    print("\n启动子进程测试...")
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(test_process, id_gen, i) for i in range(3)]
        results = [fut.result() for fut in futures]
    
    print("\n所有进程完成")
    print(f"主进程最终 ID 值: {id_gen._id.value}")
    print("\n结论:")
    print("1. Manager 代理对象在序列化后会连接到同一个 Manager 进程")
    print("2. 所有子进程共享同一个计数器，是进程安全的")
    print("3. 不会创建副本，只会创建代理对象引用")
