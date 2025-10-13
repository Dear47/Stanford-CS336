#%%
import os
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Iterator, List, BinaryIO
from tqdm import tqdm
from cs336_basics.utils.logger import get_logger
logger = get_logger(__name__, 'pretokenizer.txt')
#%%
class PreTokenizer:
    def __init__(self, special_tokens: List[str]|None=None):
        """
        初始化预处理器

        Args:
            special_tokens: 特殊token列表，如 ["", "<|endoftext|>"]
        """
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = None
        self.special_tokens_pattern = (
            "|".join(re.escape(token) for token in self.special_tokens)
            if self.special_tokens else r"(?!)"
        )

        # GPT-2 兼容的分词正则
        self.word_pattern = re.compile(
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+$|\n+"
        )
        self.logger = logger

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
        if not text:
            return []
        if self.special_tokens_pattern != r"(?!)":
            parts = re.split(f'({self.special_tokens_pattern})', text)
            tokens = []
            for part in parts:
                if part in self.special_tokens:
                    tokens.append(part.encode('utf-8'))
                elif part:
                    tokens.extend(match.group(0).encode('utf-8') for match in self.word_pattern.finditer(part))
            return tokens
        else:
            return [match.group(0).encode('utf-8') for match in self.word_pattern.finditer(text)]

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

    def read_chunks(
        self,
        input_path: str,
        split_token: str = "\n\n",
        min_chunk_size: int = 1024,  # 忽略太小的 chunk
        output_tokens = False,
    ) -> Iterator[str]:
        """
        惰性读取所有 chunks(用于单进程流式处理)

        Args:
            output_tokens: 如果为 True, 直接输出 tokens; 如果为 False, 输出文本 chunks

        Yields:
            文本 chunk 或 token (bytes)
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
                    if text:
                        if output_tokens:
                        # 直接预分词并 yield tokens
                            yield from self.pretokenize(text)
                    else:
                        # 返回文本 chunk
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
        
        # 第二步：并行处理每个 chunk
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.pretokenize,chunk) for chunk in chunks]
            iter_futures = tqdm(futures, desc="Processing Chunks", disable=not use_tqdm)

            for future in iter_futures:
                try:
                    tokens = future.result()
                    for token in tokens:
                        yield token
                except Exception as e:
                    self.logger.debug(f"Error in worker: {e}")

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
        return total_freq
    
#%%
if __name__ == '__main__':
    # preprocesser = PreTokenizer(special_tokens=["<|endoftext|>"])
    preprocesser = PreTokenizer()
    test_text = "\n\nThis is a test text, which has 换行符.\nThis is a new line."
    tokens = preprocesser.pretokenize(test_text)
    #%%
    import pathlib
    FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "tests/fixtures"
#     text = """
# Four score and seven years ago our fathers brought forth, on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.
# Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.
# But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—and that government of the people, by the people, for the people, shall not perish from the earth."""
#     token1 = preprocesser.pretokenize(text)
#     print(f"token1:{token1}")
#     print(f"length:{len(token1)}")

#     texts = [
#         'Four score and seven years ago our fathers brought forth, on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.',
#         'Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.',
#         'But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—and that government of the people, by the people, for the people, shall not perish from the earth.',
#     ]

#     tokens_iter = preprocesser.pretokenize_iter(texts)
#     length = 0
#     for tokens in tokens_iter:
#         print(tokens)
#         length += 1
#     print(f"已完成迭代")
#     print(f"length:{length}")
# %%
