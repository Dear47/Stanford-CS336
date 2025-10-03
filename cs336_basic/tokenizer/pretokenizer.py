#%%
import os
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Iterator, List, Tuple, BinaryIO
from tqdm import tqdm
from cs336_basics.utils.log import get_logger
logger = get_logger(__name__, 'pretokenizer.txt')
#%%
class PreTokenizer:
    def __init__(self, special_tokens: List[str]):
        """
        初始化预处理器

        Args:
            special_tokens: 特殊token列表，如 ["", "<|endoftext|>"]
        """
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens_pattern = (
            "|".join(re.escape(token) for token in self.special_tokens)
            if special_tokens else r"(?!)"
        )
        # GPT-2 兼容的分词正则
        self.word_pattern = re.compile(
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+$"
        )

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes
    ) -> List[int]:
        """查找基于特殊token对齐的字节边界"""
        assert isinstance(split_special_token, bytes)

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        mini_chunk_size = 4096

        for bi in range(1, len(boundaries) - 1):
            pos = boundaries[bi]
            file.seek(pos)
            while True:
                data = file.read(mini_chunk_size)
                if not data:
                    boundaries[bi] = file_size
                    break
                idx = data.find(split_special_token)
                if idx != -1:
                    boundaries[bi] = pos + idx
                    break
                pos += mini_chunk_size
                if pos >= file_size:
                    boundaries[bi] = file_size
                    break

        return sorted(set(boundaries))

    def pretokenize(self, text: str) -> List[bytes]:
        """
        将一段文本预分词为 bytes 列表

        Args:
            text: 输入字符串

        Returns:
            token 列表（每个是 bytes）
        """
        parts = re.split(f'({self.special_tokens_pattern})', text)
        tokens = []
        for part in parts:
            if part in self.special_tokens:
                pass
            elif part:
                tokens.extend(match.group(0).encode('utf-8') for match in self.word_pattern.finditer(part))
        return tokens

    def pretokenize_iter(self, texts: Iterable[str]) -> Iterator[bytes]:
        """
        惰性预分词迭代器

        Args:
            texts: 字符串可迭代对象

        Yields:
            单个 token (bytes)
        """
        for text in texts:
            yield from self.pretokenize(text)

    def _process_chunk(self, raw_text: str) -> List[bytes]:
        """
        【可被多进程调用】处理一个文本块，返回其所有 token (bytes)
        注意：必须是实例无关的方法，或定义在模块级别才能被 pickle
        """
        return self.pretokenize(raw_text)

    def read_chunks(
        self,
        input_path: str,
        split_token: str = "\n\n",
        min_chunk_size: int = 1024  # 忽略太小的 chunk
    ) -> Iterator[str]:
        """
        惰性读取所有 chunks（用于单进程流式处理）

        Yields:
            解码后的文本 chunk
        """
        split_bytes = split_token.encode('utf-8')
        with open(input_path, 'rb') as f:
            boundaries = self.find_chunk_boundaries(f, 100, split_bytes)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end - start < min_chunk_size:
                    continue
                f.seek(start)
                chunk_data = f.read(end - start)
                try:
                    text = chunk_data.decode('utf-8', errors='ignore')
                    if text.strip():
                        yield text
                except:
                    continue

    def read_tokens_parallel(
        self,
        input_path: str,
        split_token: str,
        max_workers: int = None,
        use_tqdm: bool = True
    ) -> Iterator[bytes]:
        """
        并行处理大文件，返回全局 token 流

        Args:
            input_path: 输入文件路径
            split_token: 用于切分 chunk 的分隔符
            max_workers: 进程数，默认为 CPU 核心数
            use_tqdm: 是否显示进度条

        Yields:
            所有 token (bytes)，按顺序生成
        """
        split_bytes = split_token.encode('utf-8')
        chunks = []

        # 第一步：获取所有 chunk 文本（仅主进程做 IO）
        with open(input_path, 'rb') as f:
            boundaries = self.find_chunk_boundaries(f, 100, split_bytes)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                data = f.read(end - start)
                try:
                    text = data.decode('utf-8', errors='ignore').strip()
                    if text:
                        chunks.append(text)
                except:
                    pass

        if not chunks:
            return

        # logger.debug(f"Chunk number:{len(chunks)}")

        # 第二步：并行处理每个 chunk
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.pretokenize, chunk) for chunk in chunks]
            iter_futures = tqdm(futures, desc="Processing Chunks", disable=not use_tqdm)

            for future in iter_futures:
                try:
                    tokens = future.result()
                    for token in tokens:
                        yield token
                except Exception as e:
                    print(f"Error in worker: {e}")

    def build_vocab_from_file(
        self,
        input_path: str,
        split_token: str = "<|endoftext|>",
        max_workers: int = None
    ) -> Counter:
        """
        快速从大文件构建词频表（用于后续 BPE 训练）

        Returns:
            Counter[bytes]: 每个 token 的出现频率
        """
        total_freq = Counter()
        for token in self.read_tokens_parallel(input_path, split_token, max_workers, use_tqdm=True):
            total_freq[token] += 1
        # logger.debug(f"{total_freq}")
        return total_freq
    
#%%
if __name__ == '__main__':
    import pathlib
    FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests/fixtures"
    tokenizer = PreTokenizer(special_tokens=["<|endoftext|>"])
    total_token = tokenizer.build_vocab_from_file(FIXTURES_PATH/"corpus.en",split_token='<|endoftext|>',max_workers=4)
    print(f"{total_token}")