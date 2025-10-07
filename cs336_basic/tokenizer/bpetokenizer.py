#%%
import json
from cs336_basics.tokenizer.pretokenizer import PreTokenizer
from typing import List, Dict, Tuple, Iterable
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
    with open(FIXTURES_PATH / "address.txt") as f:
        corpus_contents = f.read()

    logger.debug(f"{corpus_contents}")

    ids = tokenizer.encode(corpus_contents)

    dec_contents = tokenizer.decode(ids)

    logger.debug(f"{dec_contents}")
    assert dec_contents == corpus_contents  # 问题：缺少换行
# %%
