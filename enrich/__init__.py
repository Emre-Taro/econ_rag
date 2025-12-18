"""metadata付与の合流レイヤ（enrich）

このパッケージは、loader/chunker/embedder など既存コンポーネントを変更せずに、
実行時に注入したい metadata を Document / Chunk に統合するための薄いレイヤです。

設計方針:
- embedding処理の中では metadata を触らない
- enrich は metadata の付与/更新に専念する（LLM分類などはここでは行わない）
- 既存の metadata を壊さない（原則として未設定キーのみを埋める）
"""

from .doc_enricher import enrich_document
from .chunk_enricher import enrich_chunk

__all__ = ["enrich_document", "enrich_chunk"]


