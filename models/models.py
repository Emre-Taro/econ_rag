"""
データモデル定義

DocumentとChunkを定義するモジュール
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import hashlib


@dataclass
class Document:
    """ドキュメントを表現するクラス"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    """チャンクを表現するクラス"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:16]

