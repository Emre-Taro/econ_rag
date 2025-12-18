"""Document metadata の合流点（enricher）

このモジュールは、Document に対して「実行時に注入したい metadata」を統合するための
最小実装です。loader が付与した source/page/file_name 等の既存メタデータは尊重し、
原則として *未設定のキーのみ* を埋めます（既存値は上書きしません）。
"""

from __future__ import annotations

from typing import Any, Dict

from ..models import Document


_DOCUMENT_CONTEXT_KEYS = (
    "tenant_id",
    "index_version",
    "domain",
    "topic",
    "audience",
    "ingest_batch",
)


def _set_if_missing(metadata: Dict[str, Any], key: str, value: Any) -> None:
    """既存値を壊さずに key を埋める。

    - key が存在しない場合: セット
    - key が存在しても値が None/空文字 の場合: セット
    - それ以外: 既存値を優先（上書きしない）
    """
    if key not in metadata or metadata.get(key) in (None, ""):
        metadata[key] = value


def enrich_document(doc: Document, context: Dict[str, Any]) -> Document:
    """Document に実行時 metadata を合流させる（in-place）。

    Args:
        doc: metadata を付与する対象 Document
        context: 実行時に注入したい情報。例:
            {
              "tenant_id": "t_123",
              "index_version": "v1",
              "domain": "economics",
              "topic": "macro_model",
              "audience": "undergraduate",
              "ingest_batch": "2025-12-17T12:00:00Z"
            }

    Returns:
        doc（同じインスタンス）。metadata は in-place で更新される。
    """
    if doc.metadata is None:
        # Document は default_factory=dict なので通常 None にならないが、安全策として。
        doc.metadata = {}

    for k in _DOCUMENT_CONTEXT_KEYS:
        if k in context:
            _set_if_missing(doc.metadata, k, context[k])

    return doc


