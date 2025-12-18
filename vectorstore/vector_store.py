"""
ベクトルストアモジュール

VectorStoreを定義
"""

from typing import List, Dict, Optional
import numpy as np

from ..models import Chunk


class VectorStore:
    """ベクトルストアのベースクラス"""
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: List[np.ndarray] = []
        self.chunk_id_to_index: Dict[str, int] = {}
    
    def add_chunks(self, chunks: List[Chunk], embeddings: List[np.ndarray]):
        """
        チャンクと埋め込みを追加
        
        Args:
            chunks: 追加するChunkのリスト
            embeddings: 対応する埋め込みベクトルのリスト
        """
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        
        for i, chunk in enumerate(chunks):
            self.chunk_id_to_index[chunk.chunk_id] = start_idx + i
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """チャンクIDからチャンクを取得"""
        idx = self.chunk_id_to_index.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None
    
    def clear(self):
        """ストアをクリア"""
        self.chunks = []
        self.embeddings = []
        self.chunk_id_to_index = {}

