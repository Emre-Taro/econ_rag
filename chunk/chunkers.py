"""
チャンキングモジュール

Chunkerとその実装クラスを定義
"""

from typing import List, Dict, Any, Optional, Tuple
from ..models import Document, Chunk
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def split_paragraphs(text: str) -> List[Tuple[str, int, int]]:
    """
    段落単位に分割し、(段落テキスト, start_pos, end_pos) を返す。
    PDF抽出テキストは改行が多いので、空行で段落扱いにするのが無難。
    意味的に分けた時の最小単位に分割する事ができる。
    """
    paras = []
    start = 0
    buf = []
    i = 0
    lines = text.splitlines(keepends=True)

    def flush(buf, start_pos, end_pos):
        t = "".join(buf).strip()
        if t:
            paras.append((t, start_pos, end_pos))

    pos = 0
    cur_start = 0
    for line in lines:
        is_blank = (line.strip() == "")
        if not buf:
            cur_start = pos

        if is_blank:
            flush(buf, cur_start, pos)
            buf = []
        else:
            buf.append(line)
        pos += len(line)

    flush(buf, cur_start, pos)
    return paras


def split_by_tokens(
    text: str,
    token_counter,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    テキストをトークン数に基づいて分割する（オーバーラップ付き）
    
    Args:
        text: 分割するテキスト
        token_counter: トークン数をカウントする関数
        chunk_size: チャンクサイズ（トークン数）
        overlap: オーバーラップするトークン数
        
    Returns:
        分割されたテキストのリスト
    """
    if not text:
        return []
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        word_tokens = token_counter(word)
        
        if current_tokens + word_tokens > chunk_size and current_chunk:
            # 現在のチャンクを保存
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # オーバーラップのために最後の数単語を保持
            overlap_tokens = 0
            overlap_words = []
            for w in reversed(current_chunk):
                w_tokens = token_counter(w)
                if overlap_tokens + w_tokens <= overlap:
                    overlap_words.insert(0, w)
                    overlap_tokens += w_tokens
                else:
                    break
            
            current_chunk = overlap_words
            current_tokens = overlap_tokens
        
        current_chunk.append(word)
        current_tokens += word_tokens
    
    # 最後のチャンクを追加
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks


class Chunker(ABC):
    """チャンキングのベースクラス"""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        ドキュメントをチャンクに分割する
        
        Args:
            document: チャンキング対象のDocument
            
        Returns:
            Chunkのリスト
        """
        raise NotImplementedError("サブクラスで実装してください")


class SemanticChunker(Chunker):
    """
    段落（or 小スパン）を最小単位にして、隣接類似度が落ちる箇所でチャンク境界を作る。
    """

    def __init__(
        self,
        embedder,  # embedders.py の Embedder を想定（embed_texts(list[str]) -> list[list[float]] など）
        max_tokens: int = 700,
        min_tokens: int = 150,
        similarity_threshold: float = 0.72,
        hard_break_ratio: float = 0.15,
        token_counter=None,  # 既存のtoken数カウンタ関数を注入（text -> int）
    ):
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.similarity_threshold = similarity_threshold
        self.hard_break_ratio = hard_break_ratio
        self.token_counter = token_counter or (lambda s: len(s))  # 既存があれば置き換え

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        if not text:
            return []
        paras = split_paragraphs(text)
        print("num_paras:", len(paras))
        if not paras:
            return []

        # 段落テキストだけ抽出
        units = [p[0] for p in paras]

        # 段落embedding
        vecs = self.embedder.embed_texts(units)

        # 隣接類似度
        sims = [1.0]  # 先頭はダミー
        for i in range(1, len(vecs)):
            sims.append(cosine_sim(vecs[i-1], vecs[i]))

        chunks: List[Chunk] = []
        cur_unit_idxs: List[int] = [0]
        cur_text = units[0]
        cur_start = paras[0][1]
        cur_end = paras[0][2]

        def emit_chunk(unit_idxs: List[int], ctext: str, start_pos: int, end_pos: int, reason: str):
            total_tokens = self.token_counter(ctext)
            # --- 巨大 chunk の場合 ---
            if total_tokens > 1000:
                sub_texts = split_by_tokens(
                    ctext,
                    token_counter=self.token_counter,
                    chunk_size=500,
                    overlap=50,
                )

                for sub_i, sub_text in enumerate(sub_texts):
                    md = document.metadata.copy()
                    md.update({
                        "chunk_index": len(chunks),
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "unit_start": unit_idxs[0],
                        "unit_end": unit_idxs[-1],
                        "split_reason": "oversize_fallback",
                        "parent_chunk_tokens": total_tokens,
                        "sub_chunk_index": sub_i,
                    })
                    chunks.append(
                        Chunk(content=sub_text, metadata=md, doc_id=document.doc_id)
                    )
                return

            md = document.metadata.copy()
            md.update({
                "chunk_index": len(chunks),
                "start_pos": start_pos,
                "end_pos": end_pos,
                "unit_start": unit_idxs[0],
                "unit_end": unit_idxs[-1],
                "split_reason": reason,
                "avg_adj_similarity": (
                    sum(sims[i] for i in unit_idxs[1:]) / max(1, (len(unit_idxs) - 1))
                ),
            })
            chunks.append(Chunk(content=ctext, metadata=md, doc_id=document.doc_id))

        for i in range(1, len(units)):
            next_text = units[i]
            tentative = (cur_text + "\n\n" + next_text).strip()
            tentative_tokens = self.token_counter(tentative)

            # 話題の切れ目っぽいか？
            is_semantic_break = sims[i] < self.similarity_threshold

            # 強制的に切りたいほど低い類似度（図→説明、制度切替などでよく起きる）
            is_hard_break = sims[i] < (self.similarity_threshold * self.hard_break_ratio)

            # ルール：
            # 1) max_tokens超えるなら、そこで切る（ただし短すぎ回避）
            # 2) 意味的切れ目なら、min_tokens以上たまっていたら切る
            # 3) hard_break なら、min_tokens多少未満でも切る（ただし極端に短いのは避ける）
            cur_tokens = self.token_counter(cur_text)

            if tentative_tokens > self.max_tokens:
                emit_chunk(cur_unit_idxs, cur_text, cur_start, cur_end, reason="max_tokens")
                # reset
                cur_unit_idxs = [i]
                cur_text = next_text
                cur_start = paras[i][1]
                cur_end = paras[i][2]
                continue

            if is_hard_break and cur_tokens >= int(self.min_tokens * 0.6):
                emit_chunk(cur_unit_idxs, cur_text, cur_start, cur_end, reason="hard_semantic_break")
                cur_unit_idxs = [i]
                cur_text = next_text
                cur_start = paras[i][1]
                cur_end = paras[i][2]
                continue

            if is_semantic_break and cur_tokens >= self.min_tokens:
                emit_chunk(cur_unit_idxs, cur_text, cur_start, cur_end, reason="semantic_break")
                cur_unit_idxs = [i]
                cur_text = next_text
                cur_start = paras[i][1]
                cur_end = paras[i][2]
                continue

            # つなぐ
            cur_unit_idxs.append(i)
            cur_text = tentative
            cur_end = paras[i][2]

        # last
        emit_chunk(cur_unit_idxs, cur_text, cur_start, cur_end, reason="end_of_doc")
        return chunks
