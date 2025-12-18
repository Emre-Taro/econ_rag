"""Chunk metadata の合流点（enricher）

このモジュールは、Chunk に対して「チャンク単位で付与したい metadata」を統合するための
最小実装です。chunker が付与した chunk_index/split_reason 等や、元Document由来の metadata
は尊重し、原則として *未設定のキーのみ* を埋めます（既存値は上書きしません）。

ここでは LLM による分類などは行いません。必要であれば後からこの層に拡張できます。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..models import Chunk


_CHUNK_CONTEXT_KEYS = (
    "section",
    "content_type",
    "quality_score",
)


def _set_if_missing(metadata: Dict[str, Any], key: str, value: Any) -> None:
    """既存値を壊さずに key を埋める（None/空文字は未設定扱い）。"""
    if key not in metadata or metadata.get(key) in (None, ""):
        metadata[key] = value


def _infer_content_type(text: str) -> str:
    """簡易ルールで content_type を推定する（最小実装）。"""
    t = text or ""
    # ざっくりしたヒューリスティック
    if ("|" in t and t.count("|") >= 4) or "表" in t or "Table" in t:
        return "table"
    if "よって" in t or "したがって" in t or "また" in t or "↓" in t:
        return "procedure"
    if "例" in t:
        return "example"
    if "図" in t or "Figure" in t:
        return "figure"
    if "式" in t or any(sym in t for sym in ("∑", "∫", "√")):
        return "equation"
    if "=" in t and len(t) < 600:
        return "equation"
    if len(t.strip()) < 120:
        return "short"
    return "text"


def _infer_quality_score(text: str) -> float:
    """簡易ルールで quality_score を推定する（0.0-1.0, 最小実装）。"""
    t = (text or "").strip()
    if not t:
        return 0.0

    score = 0.6

    # 短すぎるチャンクは情報量が少ない可能性
    if len(t) < 80:
        score -= 0.25

    # 文字化け/置換文字が多い場合は品質を下げる
    if "\ufffd" in t:
        score -= 0.25

    # 極端に長い場合（ノイズ混入の可能性）
    if len(t) > 5000:
        score -= 0.15

    # 0.0-1.0 にクランプ
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)


def enrich_chunk(chunk: Chunk, context: Dict[str, Any]) -> Chunk:
    """Chunk にチャンク単位 metadata を合流させる（in-place）。

    Args:
        chunk: metadata を付与する対象 Chunk
        context: 実行時に注入したい情報。例:
            {
              "section": "2.3 IS-LMの導出",
              "content_type": "text",
              "quality_score": 0.9
            }
            省略したキーは簡易ルールで補完されます（content_type, quality_score）。

    Returns:
        chunk（同じインスタンス）。metadata は in-place で更新される。
    """
    if chunk.metadata is None:
        chunk.metadata = {}

    # context 由来（既存値は上書きしない）
    for k in _CHUNK_CONTEXT_KEYS:
        if k in context:
            _set_if_missing(chunk.metadata, k, context[k])

    # ルール由来の補完（contextが無い場合のみ）
    if "content_type" not in chunk.metadata or chunk.metadata.get("content_type") in (None, ""):
        _set_if_missing(chunk.metadata, "content_type", _infer_content_type(chunk.content))

    if "quality_score" not in chunk.metadata or chunk.metadata.get("quality_score") in (None, ""):
        _set_if_missing(chunk.metadata, "quality_score", _infer_quality_score(chunk.content))

    # section は推定ロジックを入れていない（最小実装）。必要なら後から拡張。
    return chunk


