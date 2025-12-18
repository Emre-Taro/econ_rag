"""
RAG Pipeline - 拡張可能なベース実装

このパイプラインは、以下のコンポーネントで構成されています：
- DocumentLoader: ドキュメント読み込み
- Chunker: テキストチャンキング
- Embedder: 埋め込み生成
- VectorStore: ベクトルストア
- Retriever: 検索・リトリーバル
- RAGPipeline: 全体統合

各コンポーネントは拡張可能な設計になっており、
メタデータ設計、チャンク戦略、検索戦略、rerank、フィルタを
後から追加・カスタマイズできます。
"""

# データモデル
from .models import Document, Chunk

# ローダー
from .loading import (
    DocumentLoader,
    SimpleTextLoader,
    PDFLoader,
    AutoLoader
)

# チャンカー
from .chunk import (
    SemanticChunker
)

# 埋め込み
from .embedding import (
    Embedder,
    DummyEmbedder,
    OpenAIEmbedder
)

# ベクトルストア
from .vectorstore import VectorStore

# リトリーバー
from .retrievers import (
    Retriever,
    SimpleRetriever
)

# パイプライン
from .pipeline import RAGPipeline

__all__ = [
    # データモデル
    'Document',
    'Chunk',
    # ローダー
    'DocumentLoader',
    'SimpleTextLoader',
    'PDFLoader',
    'AutoLoader',
    # チャンカー
    'SemanticChunker',
    # 埋め込み
    'Embedder',
    'DummyEmbedder',
    'OpenAIEmbedder',
    # ベクトルストア
    'VectorStore',
    # リトリーバー
    'Retriever',
    'SimpleRetriever',
    # パイプライン
    'RAGPipeline',
]

