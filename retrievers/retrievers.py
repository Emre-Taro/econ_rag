"""
検索・リトリーバルモジュール

Retrieverとその実装クラスを定義
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..models import Chunk
from ..vectorstore import VectorStore
from ..embedding import Embedder


class Retriever:
    """検索・リトリーバルのベースクラス（拡張可能）"""
    
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        """
        Args:
            vector_store: ベクトルストア
            embedder: 埋め込み生成器
        """
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        クエリに対して関連するチャンクを検索
        
        Args:
            query: 検索クエリ
            top_k: 取得するチャンク数
            filters: フィルタ条件（拡張用、現在は未使用）
            
        Returns:
            (Chunk, score)のタプルのリスト
        """
        raise NotImplementedError("サブクラスで実装してください")
    
    def apply_filters(
        self,
        chunks: List[Chunk],
        filters: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        フィルタを適用（拡張用のメソッド）
        
        Args:
            chunks: フィルタ対象のチャンク
            filters: フィルタ条件
            
        Returns:
            フィルタ後のチャンクリスト
        """
        if filters is None:
            return chunks
        
        # ベース実装：メタデータでのフィルタリング例
        filtered = chunks
        for key, value in filters.items():
            filtered = [
                chunk for chunk in filtered
                if chunk.metadata.get(key) == value
            ]
        return filtered
    
    def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        再ランキング（拡張用のメソッド）
        
        Args:
            query: クエリ
            chunks_with_scores: チャンクとスコアのタプルリスト
            top_k: 再ランキング後の取得数（Noneの場合は全て）
            
        Returns:
            再ランキング後のチャンクとスコアのリスト
        """
        # ベース実装：そのまま返す（拡張可能）
        if top_k is not None:
            return chunks_with_scores[:top_k]
        return chunks_with_scores


# def _infer_content_type_from_query(query: str) -> Optional[str]:
#     """クエリから content_type を推論する
    
#     Args:
#         query: 検索クエリ
        
#     Returns:
#         推論されたcontent_type（Noneの場合はフィルタリングしない）
#     """
#     q = (query or "").lower()
    
#     # 具体例を求めている場合
#     example_keywords = [
#         "具体例", "例", "example", "例えば", "例を", "例を挙げて", 
#         "例を示して", "例を教えて", "サンプル", "sample"
#     ]
#     if any(kw in q for kw in example_keywords):
#         return "example"
    
#     # 定義を求めている場合
#     definition_keywords = [
#         "定義", "definition", "とは", "とは何", "意味", "意味は", 
#         "説明して", "説明", "何か", "何", "what is", "what's"
#     ]
#     if any(kw in q for kw in definition_keywords):
#         return "short"
    
#     # 論理・手順・プロセスについて聞いている場合
#     procedure_keywords = [
#         "論理", "logic", "手順", "プロセス", "process", "流れ", 
#         "導出", "derivation", "なぜ", "理由", "reason", "how", 
#         "どのように", "どうやって", "方法", "method", "よって", 
#         "したがって", "therefore", "thus", "step", "ステップ"
#     ]
#     if any(kw in q for kw in procedure_keywords):
#         return "procedure"
    
#     # マッチしない場合はNoneを返す（フィルタリングしない）
#     return None


class SimpleRetriever(Retriever):
    """シンプルなコサイン類似度ベースのリトリーバー"""
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        embedder: Embedder,
        min_quality_score: float = 0.4,
        min_avg_adj_similarity: float = 0.3,
        excluded_page_numbers: Optional[List[int]] = None
    ):
        """
        Args:
            vector_store: ベクトルストア
            embedder: 埋め込み生成器
            min_quality_score: 最小quality_score閾値（デフォルト: 0.6）
            min_avg_adj_similarity: 最小avg_adj_similarity閾値（デフォルト: 0.4）
            excluded_page_numbers: 除外するページ番号のリスト（デフォルト: [7, 10]）
        """
        super().__init__(vector_store, embedder)
        self.min_quality_score = min_quality_score
        self.min_avg_adj_similarity = min_avg_adj_similarity
        self.excluded_page_numbers = excluded_page_numbers if excluded_page_numbers is not None else [7, 10]
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        return_filter_stats: bool = False
    ) -> List[Tuple[Chunk, float]] | Tuple[List[Tuple[Chunk, float]], Dict[str, Any]]:
        """コサイン類似度で検索し、フィルタリングを適用
        
        Args:
            query: 検索クエリ
            top_k: 取得するチャンク数
            filters: 追加のフィルタ条件
            return_filter_stats: Trueの場合、フィルタリング統計情報も返す
            
        Returns:
            フィルタリング後の結果（return_filter_stats=Trueの場合は統計情報も含む）
        """
        if len(self.vector_store.chunks) == 0:
            if return_filter_stats:
                return [], {
                    "total_before_filter": 0,
                    "filtered_by_quality_score": 0,
                    "filtered_by_avg_adj_similarity": 0,
                    "filtered_by_content_type": 0,
                    "filtered_by_page_number": 0,
                    "total_after_filter": 0,
                    # "inferred_content_type": None,
                    "active_filters": []
                }
            return []
        
        # クエリを埋め込み
        query_embedding = self.embedder.embed(query)
        
        # コサイン類似度を計算
        scores = []
        for embedding in self.vector_store.embeddings:
            # コサイン類似度
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            scores.append(similarity)
        
        # スコアでソート
        indices = np.argsort(scores)[::-1]  # 降順
        
        # チャンクとスコアを取得
        results = [
            (self.vector_store.chunks[i], float(scores[i]))
            for i in indices
        ]
        
        # デフォルトフィルタリングを適用
        results, filter_stats = self._apply_default_filters(query, results)
        
        # 追加のフィルタ適用（拡張用）
        if filters:
            filtered_results = []
            for chunk, score in results:
                if self._matches_filters(chunk, filters):
                    filtered_results.append((chunk, score))
            results = filtered_results
            filter_stats["total_after_filter"] = len(results)
        
        final_results = results[:top_k]
        
        if return_filter_stats:
            return final_results, filter_stats
        return final_results
    
    def _apply_default_filters(
        self,
        query: str,
        results: List[Tuple[Chunk, float]]
    ) -> Tuple[List[Tuple[Chunk, float]], Dict[str, Any]]:
        """デフォルトのフィルタリングを適用
        
        - quality_score >= min_quality_score (デフォルト: 0.6)
        - avg_adj_similarity >= min_avg_adj_similarity (デフォルト: 0.4)
        - content_type: クエリから推論（該当する場合のみ）
        - page_number: excluded_page_numbers を除外（デフォルト: [7, 10]）
        
        Returns:
            (フィルタ後の結果, フィルタリング統計情報)
        """
        filtered = []
        
        # フィルタリング統計情報
        stats = {
            "total_before_filter": len(results),
            "filtered_by_quality_score": 0,
            "filtered_by_avg_adj_similarity": 0,
            "filtered_by_content_type": 0,
            "filtered_by_page_number": 0,
            "total_after_filter": 0,
            # "inferred_content_type": None,
            "active_filters": []
        }
        
        # # クエリからcontent_typeを推論
        # inferred_content_type = _infer_content_type_from_query(query)
        # stats["inferred_content_type"] = inferred_content_type
        
        # if inferred_content_type is not None:
        #     stats["active_filters"].append(f"content_type={inferred_content_type}")
        
        stats["active_filters"].extend([
            f"quality_score>={self.min_quality_score}",
            f"avg_adj_similarity>={self.min_avg_adj_similarity}",
            f"page_number NOT IN {self.excluded_page_numbers}"
        ])
        
        for chunk, score in results:
            metadata = chunk.metadata or {}
            filtered_out = False
            
            # quality_score フィルタリング
            quality_score = metadata.get("quality_score")
            if quality_score is not None:
                try:
                    quality_score = float(quality_score)
                    if quality_score < self.min_quality_score:
                        stats["filtered_by_quality_score"] += 1
                        filtered_out = True
                        continue
                except (ValueError, TypeError):
                    # quality_scoreが数値でない場合はスキップ（フィルタリングしない）
                    pass
            
            # avg_adj_similarity フィルタリング
            avg_adj_similarity = metadata.get("avg_adj_similarity")
            if avg_adj_similarity is not None:
                try:
                    avg_adj_similarity = float(avg_adj_similarity)
                    if avg_adj_similarity < self.min_avg_adj_similarity:
                        stats["filtered_by_avg_adj_similarity"] += 1
                        filtered_out = True
                        continue
                except (ValueError, TypeError):
                    # avg_adj_similarityが数値でない場合はスキップ（フィルタリングしない）
                    pass
            
            # # content_type フィルタリング（クエリから推論された場合のみ）
            # if inferred_content_type is not None:
            #     chunk_content_type = metadata.get("content_type")
            #     if chunk_content_type != inferred_content_type:
            #         stats["filtered_by_content_type"] += 1
            #         filtered_out = True
            #         continue
            
            # page_number フィルタリング
            page_number = metadata.get("page_number")
            if page_number is not None:
                try:
                    page_number = int(page_number)
                    if page_number in self.excluded_page_numbers:
                        stats["filtered_by_page_number"] += 1
                        filtered_out = True
                        continue
                except (ValueError, TypeError):
                    # page_numberが数値でない場合はスキップ（フィルタリングしない）
                    pass
            
            if not filtered_out:
                filtered.append((chunk, score))
        
        stats["total_after_filter"] = len(filtered)
        return filtered, stats
    
    def _matches_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
        """フィルタ条件に一致するかチェック"""
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True

