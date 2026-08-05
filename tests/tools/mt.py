import os
import threading
import argparse
from tqdm import tqdm

def copy_block(src_path, dst_path, offset, size, progress):
    """拷贝文件的一块，并更新进度条"""
    with open(src_path, 'rb') as f_src, open(dst_path, 'r+b') as f_dst:
        f_src.seek(offset)
        f_dst.seek(offset)
        remaining = size
        chunk_size = 16 * 1024 * 1024  # 1MB 小块拷贝
        while remaining > 0:
            read_size = min(chunk_size, remaining)
            data = f_src.read(read_size)
            f_dst.write(data)
            remaining -= read_size
            progress.update(len(data))

def multithread_copy(src_path, dst_path, num_threads=4):
    file_size = os.path.getsize(src_path)

    # 创建目标文件，并预分配大小
    with open(dst_path, 'wb') as f:
        f.truncate(file_size)

    block_size = file_size // num_threads
    threads = []

    # 进度条
    progress = tqdm(total=file_size, unit='B', unit_scale=True, desc="拷贝进度")

    for i in range(num_threads):
        offset = i * block_size
        size = block_size if i < num_threads - 1 else file_size - offset
        t = threading.Thread(target=copy_block, args=(src_path, dst_path, offset, size, progress))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    progress.close()
    print(f"文件拷贝完成: {src_path} -> {dst_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多线程拷贝大文件（带进度条）")
    parser.add_argument("src", help="源文件路径")
    parser.add_argument("dst", help="目标文件路径")
    parser.add_argument("-t", "--threads", type=int, default=4, help="线程数，默认4")

    args = parser.parse_args()
    multithread_copy(args.src, args.dst, args.threads)
