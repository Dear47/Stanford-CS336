#%%
import json
import collections
import regex as re
from cs336_basics.tokenizer.pretokenizer import PreTokenizer
from typing import List, Dict, Tuple, Set
from collections import Counter
import heapq
from cs336_basics.utils.logger import get_logger
from tests.common import gpt2_bytes_to_unicode
logger = get_logger(__name__, 'trainer.txt')
#%%
class ReversedBytesPair:
    """
    一个反转比较的包装类, 用于实现最大堆
    """
    __slots__ = ('pair',)
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair  # 反转比较
    def __eq__(self, other):
        return self.pair == other.pair

class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        """
        初始化BPE训练器
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.preprocessor = PreTokenizer(special_tokens)
        self.vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.pair_freq: Dict[Tuple[bytes, bytes], int] = {}  # 记录每一个字节对的频率(不断更新)
        self.splits: Dict[bytes, list[bytes]] = {}  # 将每一个token拆分成一个一个symbols
        self.token_freq: Dict[list[bytes], int] = {}
        self.pair_to_token: Dict[Tuple[bytes, bytes], Set[bytes]] = {}  # 记录每一个字节对所对应的token(不断更新)
        self.freq_max_heap = []
        self.logger = logger

    def _push_pair_to_heap(self, pair:Tuple[bytes, bytes], freq:int):
        """
        按负频率和反转比较后的字典序构建最大堆
        """
        if freq <= 0:
            return  # 忽略无效频率
        heapq.heappush(self.freq_max_heap, (-freq, ReversedBytesPair(pair), pair))

    def _pop_pair_from_heap(self)->Tuple[bytes, bytes]:
        while self.freq_max_heap:
            neg_freq,_,pair = heapq.heappop(self.freq_max_heap)
            freq = -neg_freq
            if pair in self.pair_freq and self.pair_freq[pair] == freq:
                return pair
        raise ValueError("堆没有返回频率最大的字节对")
    
    def _update_pair_freq(self,
                          new_pair_part:Tuple[bytes,bytes],
                          old_pair_part:Tuple[bytes,bytes],
                          token:bytes,
                          freq:int):
        """
        由于best_pair合并, 会在相关token中产生new_pair, 同时old_pair会消失, 应该考虑这两者的变化
        """
        self.pair_to_token.setdefault(new_pair_part,set()).add(token)
        self.pair_freq[new_pair_part] = self.pair_freq.get(new_pair_part,0)+freq  # freq是受影响的token的频率
        self._push_pair_to_heap(new_pair_part,self.pair_freq[new_pair_part])

        if old_pair_part in self.pair_freq:
            self.pair_freq[old_pair_part] -= freq
            if self.pair_freq[old_pair_part] <= 0:
                del self.pair_freq[old_pair_part]
                del self.pair_to_token[old_pair_part]
            else:
                self._push_pair_to_heap(old_pair_part,self.pair_freq[old_pair_part])

    def initialize_splits_and_pairs(self, token_freq:Counter):
        """
        根据pretokenizer提供的token_freq初始化splits、pair_freq、pair_to_token
        """
        for token, freq in token_freq.items():
            self.splits[token] = [bytes([b]) for b in token]
            token_pieces = self.splits[token]
            if len(token_pieces) == 1:
                continue
            
            for pair in zip(token_pieces[:-1],token_pieces[1:]):
                self.pair_freq[pair] = self.pair_freq.get(pair,0) + freq

                if pair not in self.pair_to_token:
                    self.pair_to_token[pair] = set()
                self.pair_to_token[pair].add(token)

        self.pair_freq = {pair:freq for pair,freq in self.pair_freq.items() if freq > 0}

        for pair, freq in self.pair_freq.items():
            self._push_pair_to_heap(pair,freq)
        
    def find_best_pair(self)->Tuple[bytes,bytes]:
        return self._pop_pair_from_heap()

    def add_special_token(self):
        """
        将special_tokens加入vocab中
        """
        for idx, token in enumerate(self.special_tokens):
            self.vocab[self.vocab_size-len(self.special_tokens)+idx] = token.encode('utf-8')
            
    def merge(self,best_pair:Tuple[bytes,bytes],new_token:bytes,token_freq:Counter):
        """
        合并字节对, 并更新pair_freq
        """
        affected_tokens = list(self.pair_to_token.get(best_pair, set()))
        
        if best_pair in self.pair_freq:
            # 删除pair_freq中的best_pair
            del self.pair_freq[best_pair]
        if best_pair in self.pair_to_token:
            del self.pair_to_token[best_pair]

        for token in affected_tokens:
            freq = token_freq[token]
            token_pieces = self.splits[token]
            i = 0
            while i < len(token_pieces) -1 :
                if token_pieces[i] == best_pair[0] and token_pieces[i+1] == best_pair[1]:
                    token_pieces[i] = new_token
                    token_pieces.pop(i+1)

                    if i>0:
                        new_pair_left = (token_pieces[i-1],new_token)
                        old_pair_left = (token_pieces[i-1],best_pair[0])
                        self._update_pair_freq(new_pair_left,old_pair_left,token,freq)
                    if i<len(token_pieces)-1:
                        new_pair_right = (new_token, token_pieces[i+1])
                        old_pair_right = (best_pair[1], token_pieces[i+1])
                        self._update_pair_freq(new_pair_right,old_pair_right,token,freq)
                else:
                    i += 1

    def train(self, input_path:str)->Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        训练BPE
        """
        # 读取语料
        token_freq:Counter[bytes,int] = self.preprocessor.build_vocab_from_file(input_path=input_path,split_token="<|endoftext|>",max_workers=4)
        bytes_special_tokens = [special_token.encode('utf-8') for special_token in self.special_tokens]
        self.logger.debug(f"{bytes_special_tokens}")
        for token in bytes_special_tokens:
            if token in token_freq:
                del token_freq[token]
        
        # 开始训练
        self.vocab = {i:bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        self.merges = []

        self.initialize_splits_and_pairs(token_freq)

        for num_merge in range(num_merges):
            if not self.pair_freq:
                break

            best_pair = self.find_best_pair()  # best_pair:byte|unicode str
            self.merges.append(best_pair)

            new_token = best_pair[0]+best_pair[1]
            self.vocab[256+num_merge] = new_token
            
            self.merge(best_pair,  new_token, token_freq)

        self.add_special_token()
        return self.vocab, self.merges

def save_file(vocab:Dict[int,bytes],merges:List[Tuple[bytes,bytes]],vocab_filepath:str,merges_filepath:str):
    byte_encoder = gpt2_bytes_to_unicode()
    def encode_bytes(b:bytes)->str:
        return ''.join(byte_encoder[byte] for byte in b)
    vocab_json = {
        encode_bytes(token): token_id 
        for token_id, token in vocab.items()
    }
    with open(vocab_filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    with open(merges_filepath,'w',encoding='utf-8') as f:
        for first,second in merges:
            left = encode_bytes(first)
            right = encode_bytes(second)
            f.write(f"{left} {right}\n")

def train_bpe(input_path:str,vocab_size:int,special_tokens:List[str]):
    import pathlib
    CURRENT_PATH = pathlib.Path(__file__).resolve().parent
    tokenizer = BPETrainer(vocab_size,special_tokens)
    vocab, merges = tokenizer.train(input_path)
    vocab_filepath = CURRENT_PATH/'tinystories_vocab.json'
    merges_filepath = CURRENT_PATH/'tinystories_merges.txt'
    save_file(vocab,merges,vocab_filepath,merges_filepath)
    return vocab, merges

#%%
if __name__ == '__main__':
    import pathlib
    FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "tests/fixtures"
    # input_path = FIXTURES_PATH/"corpus.en"
    input_path = FIXTURES_PATH/"tinystories_sample.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)