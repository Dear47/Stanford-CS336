#%%
import os
import sys
import psutil
import numpy as np
import tqdm
import json
from typing import List, Dict, Tuple, Iterable, Union
from cs336_basics.tokenizer.pretokenizer import PreTokenizer
from cs336_basics.utils.log import get_logger
logger = get_logger(__name__,'bpetokenzier.txt')
from tests.common import gpt2_bytes_to_unicode
#%%
class BPETokenizer:
    def __init__(self, vocab:Dict[int, bytes], merges:List[Tuple[bytes,bytes]], special_tokens:List[str]|None=None):
        """
        初始化BPE分词器
        """
        self.merges: List[Tuple[bytes, bytes]] = merges

        self.encoder: Dict[bytes, int] = {v:k for k,v in vocab.items()}
        self.decoder: Dict[int, bytes] = vocab

        self.byte_encoder: Dict[int, str] = gpt2_bytes_to_unicode()
        self.byte_decoder: Dict[str, int] = {v:k for k,v in self.byte_encoder.items()}

        # logger.debug(f"Special Tokens: {[special_token for special_token in special_tokens]}")
        self.preprocesser = PreTokenizer(special_tokens)
        self.cache: Dict[bytes,List[int]] = {}  # 有待优化
    
    @classmethod
    def from_files(cls, vocab_path:str, merges_path:str, special_tokens:List[str]|None=None):
        byte_encoder: Dict[int, str] = gpt2_bytes_to_unicode()
        byte_decoder: Dict[str, int] = {v:k for k,v in byte_encoder.items()}

        merges:List[Tuple[bytes,bytes]] = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(' ')) == 2:
                    merges.append(
                        (
                            bytes([byte_decoder[token] for token in cleaned_line.split(' ')[0]]),
                            bytes([byte_decoder[token] for token in cleaned_line.split(' ')[1]])
                        )
                    )
        
        with open(vocab_path, 'r', encoding="utf-8") as f:
            origin_vocab = json.load(f)
        vocab = {
            index:bytes([byte_decoder[token] for token in item])
            for item, index in origin_vocab.items()
        }
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode('utf-8')
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token
        return cls(vocab, merges, special_tokens)
    
    def pretokenizer(self, text:str)->List[bytes]:
        tokens = self.preprocesser.pretokenize(text)
        return tokens
    
    def pretokenizer_iter(self, texts:Iterable[str])->Iterable[bytes]:
        tokens_iter = self.preprocesser.pretokenize_iter(texts)
        return tokens_iter
    
    def get_token_id(self, token:bytes)->List[int]:
        """
        将bytes类型的token编码成vocab中的id
        """
        origin_token = token
        # logger.debug(f"origin token:{origin_token}")
        token_id:List[int] = []
        if token in self.cache:
            "如果token在缓存中, 直接获取其id"
            token_id.extend(self.cache[token])
            return token_id
        if token in self.encoder:
            "如果是单字符token或者特殊token,直接在vocab中寻找"
            id = self.encoder[token]
            token_id.append(id)
            return token_id
        else:
            token:List[bytes] = [bytes([b]) for b in token]
            while len(token)>1:
                "对于多字符token合并, 优先采取merges中靠前的pair"
                pairs = zip(token[:-1],token[1:])
                min_rank = float('inf')
                best_index = -1
                best_pair = None
                for i, pair in enumerate(pairs):
                    if pair in self.merges:
                        rank = self.merges.index(pair)
                        if rank<min_rank:
                            min_rank = rank
                            best_index = i
                            best_pair = pair
                if best_pair is None:
                    "未发生merge,退出对此token的处理"
                    break
                token = token[:best_index]+[best_pair[0]+best_pair[1]]+token[best_index+2:]
            for t in token:
                try:
                    id = self.encoder[t]
                    token_id.append(id)
                except KeyError:
                    logger.debug(f"Token {token} not found in Vocab!")
            self.cache[origin_token] = token_id  # 放入缓存
        return token_id

    def encode(self, text:str)->List[int]:
        """
        将文本编码成BPE token id列表
        """
        tokens:List[bytes] = self.pretokenizer(text)
        # logger.debug(f"PreTokens:{tokens}")
        id_list:List[int] = []
        for token in tokens:
            token_id = self.get_token_id(token)
            id_list += token_id
            # logger.debug(f"Encode Token {token} to ID {token_id}")
        return id_list

    def encode_iterable(self, iterable:Iterable[str])->Iterable[int]:
        tokens_iter = self.pretokenizer_iter(iterable)
        for token in tokens_iter:
            token_id = self.get_token_id(token)
            yield from token_id
        
    def decode(self, ids:List[int])->str:
        bytes_text = b""
        # logger.debug(f"-"*60)
        for id in ids:
            if id in self.decoder:
                bytes_token = self.decoder[id]
                bytes_text+= bytes_token
                # logger.debug(f"Decode ID {id} to Token {bytes_token}")
            else:
                logger.debug(f"ID {id} is not in Vocab!")
        # logger.debug(f"decode {ids} to bytes_text {bytes_text}")
        return bytes_text.decode('utf-8',errors='replace')
        
    def encode_to_npfile(self, input_path: os.PathLike, output_path: os.PathLike, 
                        memory_threshold_ratio: float = 0.5) -> None:
        """
        智能保存方法：根据数据大小和可用内存自动选择保存策略
        
        Parameters:
        input_path: 输入文件路径
        output_path: 输出文件路径
        memory_threshold_ratio: 内存使用阈值比例(默认0.5, 即50%可用内存)
        """
        
        # 1. 估算token数量和所需内存
        estimated_tokens, file_size_mb = self._estimate_token_count(input_path)
        
        # uint16每个token占用2字节
        estimated_memory_bytes = estimated_tokens * 2  # uint16 = 2 bytes
        estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
        
        logger.debug(f"File Size: {file_size_mb:.2f} MB")
        logger.debug(f"Estimated Token Number: {estimated_tokens:,}")
        logger.debug(f"Estimated Memory Usage: {estimated_memory_mb:.2f} MB")
        
        # 2. 检查可用内存
        available_memory = psutil.virtual_memory().available
        available_memory_mb = available_memory / (1024 * 1024)
        max_allowed_memory = available_memory * memory_threshold_ratio
        
        logger.debug(f"Avaiable Memory: {available_memory_mb:.2f} MB")
        logger.debug(f"Max Allowed Memory: {max_allowed_memory / (1024 * 1024):.2f} MB")
        
        # 3. 选择保存策略
        if estimated_memory_bytes <= max_allowed_memory:
            logger.debug("Save directly(sufficient memory)...")
            self._save_direct_method(input_path, output_path, estimated_tokens)
        else:
            logger.debug("Save streamingly(out of memory)...")
            self._save_streaming_method(input_path, output_path)

    def _estimate_token_count(self, input_path: os.PathLike) -> tuple[int, float]:
        """估算token数量和文件大小"""
        file_size = os.path.getsize(input_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # 简单估算：假设平均每行产生约50个tokens（可根据实际情况调整）
        # 或者更保守的估算：每字节文本产生0.5-2个tokens
        with open(input_path, 'r', encoding='utf-8') as f:
            # 快速采样前1000行来估算平均tokens/行
            sample_lines = []
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                sample_lines.append(line)
        
        if sample_lines:
            sample_tokens = 0
            for token_id in self.encode_iterable(iter(sample_lines)):
                sample_tokens += 1
            
            avg_tokens_per_line = sample_tokens / len(sample_lines)
            
            # 计算总行数
            with open(input_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            estimated_tokens = int(avg_tokens_per_line * total_lines)
        else:
            # 如果文件很小，直接按文件大小估算
            # 假设平均每字节产生1个token（保守估计）
            estimated_tokens = file_size
        
        # 设置最小值和最大值保护
        estimated_tokens = max(estimated_tokens, 1000)  # 至少1000 tokens
        estimated_tokens = min(estimated_tokens, 10**9)  # 最多10亿 tokens
        
        return estimated_tokens, file_size_mb

    def _save_direct_method(self, input_path: os.PathLike, output_path: os.PathLike, 
                        expected_tokens: int = None) -> None:
        """直接保存方法 - 适用于内存充足的情况"""
        token_ids = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for token_id in tqdm(self.encode_iterable(f), desc="Encoding ...", 
                            total=expected_tokens, unit="token"):
                # 验证token ID范围
                if token_id < 0 or token_id > 65535:
                    raise ValueError(f"Token ID {token_id} is out of uint16 range:[0, 65535]")
                token_ids.append(token_id)
        
        # 直接保存为uint16数组
        np.save(output_path, np.array(token_ids, dtype=np.uint16))
        logger.debug(f"Save File to: {output_path}\nToTal Token Number: {len(token_ids)}")

    def _save_streaming_method(self, input_path: os.PathLike, output_path: os.PathLike) -> None:
        """流式保存方法 - 适用于内存不足的情况"""
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            token_count = 0
            
            with open(input_path, 'r', encoding='utf-8') as f:
                for token_id in tqdm(self.encode_iterable(f), desc="Encoding...", 
                                unit="token", total=None):
                    # 验证token ID范围
                    if token_id < 0 or token_id > 65535:
                        raise ValueError(f"Token ID {token_id} is out of uint16 range:[0, 65535]")
                    tmpfile.write(np.uint16(token_id).tobytes())
                    token_count += 1
            
            tmpfile_path = tmpfile.name
        
        # 读取临时文件创建memmap，并保存为.npy
        logger.debug("Save to the file...")
        try:
            mm_array = np.memmap(tmpfile_path, dtype=np.uint16, mode='r', shape=(token_count,))
            np.save(output_path, mm_array)
            del mm_array
        finally:
            os.remove(tmpfile_path)
        
        logger.debug(f"Save File to: {output_path}\nToTal Token Number: {token_count}")
#%%
if __name__ == '__main__':
    import pathlib
    FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent)/"tests/fixtures"
    VOCAB_PATH = FIXTURES_PATH/'gpt2_vocab.json'
    MERGES_PATH = FIXTURES_PATH/'gpt2_merges.txt'
    tokenizer = BPETokenizer.from_files(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
