"""
検索・リトリーバルモジュール

Retrieverとその実装クラスを定義
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..models import Chunk
from ..vectorstore import VectorStore
from ..embedding import Embedder

# クロスエンコーダのインポート（オプショナル）
# コメントアウト: rerankは処理が重く改善が少ないため無効化
# try:
#     from sentence_transformers import CrossEncoder
#     CROSS_ENCODER_AVAILABLE = True
# except ImportError:
#     CROSS_ENCODER_AVAILABLE = False
#     CrossEncoder = None
CROSS_ENCODER_AVAILABLE = False
CrossEncoder = None


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
        
        サブクラスで実装してください。再ランキングを実装しない場合は、
        そのまま返すか、top_kでスライスするだけの実装でも構いません。
        
        Args:
            query: クエリ
            chunks_with_scores: チャンクとスコアのタプルリスト
            top_k: 再ランキング後の取得数（Noneの場合は全て）
            
        Returns:
            再ランキング後のチャンクとスコアのリスト
        """
        # デフォルト実装：再ランキングを行わず、top_kでスライスするだけ
        # サブクラスでオーバーライドして、実際の再ランキングを実装してください
        if top_k is not None:
            return chunks_with_scores[:top_k]
        return chunks_with_scores



class SimpleRetriever(Retriever):
    """シンプルなコサイン類似度ベースのリトリーバー"""
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        embedder: Embedder,
        min_quality_score: float = 0.4,
        min_avg_adj_similarity: float = 0.3,
        excluded_page_numbers: Optional[List[int]] = None,
        use_rerank: bool = False,  # デフォルトで無効化
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_k: int = 5,
        initial_retrieval_top_n: int = 15
    ):
        """
        Args:
            vector_store: ベクトルストア
            embedder: 埋め込み生成器
            min_quality_score: 最小quality_score閾値（デフォルト: 0.4）
            min_avg_adj_similarity: 最小avg_adj_similarity閾値（デフォルト: 0.3）
            excluded_page_numbers: 除外するページ番号のリスト（デフォルト: [7, 10]）
            use_rerank: 再ランキングを使用するか（デフォルト: False、現在は無効化）
            rerank_model_name: クロスエンコーダモデル名（現在は使用されていません）
            rerank_top_k: 再ランキング後の取得数（現在は使用されていません）
            initial_retrieval_top_n: 初期検索で取得するチャンク数（現在は使用されていません）
        """
        super().__init__(vector_store, embedder)
        self.min_quality_score = min_quality_score
        self.min_avg_adj_similarity = min_avg_adj_similarity
        self.excluded_page_numbers = excluded_page_numbers if excluded_page_numbers is not None else [7, 10]
        self.use_rerank = use_rerank
        self.rerank_top_k = rerank_top_k
        self.initial_retrieval_top_n = initial_retrieval_top_n
        
        # クロスエンコーダの初期化（コメントアウト: rerankは処理が重く改善が少ないため無効化）
        self.cross_encoder = None
        # if use_rerank:
        #     if not CROSS_ENCODER_AVAILABLE:
        #         print("警告: sentence-transformersがインストールされていません。再ランキングは無効化されます。")
        #         print("インストール方法: pip install sentence-transformers")
        #         self.use_rerank = False
        #     else:
        #         try:
        #             self.cross_encoder = CrossEncoder(rerank_model_name)
        #         except Exception as e:
        #             print(f"警告: クロスエンコーダの初期化に失敗しました: {str(e)}")
        #             print("再ランキングは無効化されます。")
        #             self.use_rerank = False
    
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
        
        # 再ランキングを適用（コメントアウト: rerankは処理が重く改善が少ないため無効化）
        # if self.use_rerank and self.cross_encoder is not None:
        #     # 初期検索で topN を取得
        #     initial_results = results[:self.initial_retrieval_top_n]
        #     # 再ランキングで topK に絞る
        #     results = self.rerank(query, initial_results, top_k=self.rerank_top_k)
        #     if return_filter_stats:
        #         filter_stats["total_after_rerank"] = len(results)
        # else:
        #     # 再ランキングを使用しない場合は、指定された top_k で取得
        #     final_results = results[:top_k]
        #     if return_filter_stats:
        #         return final_results, filter_stats
        #     return final_results
        
        # 再ランキングを使用しない場合は、指定された top_k で取得
        final_results = results[:top_k]
        if return_filter_stats:
            return final_results, filter_stats
        return final_results
    
    # 再ランキングメソッド（コメントアウト: rerankは処理が重く改善が少ないため無効化）
    # def rerank(
    #     self,
    #     query: str,
    #     chunks_with_scores: List[Tuple[Chunk, float]],
    #     top_k: Optional[int] = None
    # ) -> List[Tuple[Chunk, float]]:
    #     """
    #     再ランキング（クロスエンコーダを使用）
    #     
    #     Args:
    #         query: クエリ
    #         chunks_with_scores: チャンクとスコアのタプルリスト
    #         top_k: 再ランキング後の取得数（Noneの場合は全て）
    #         
    #     Returns:
    #         再ランキング後のチャンクとスコアのリスト
    #     """
    #     if self.use_rerank and self.cross_encoder is not None:
    #         return self._rerank_with_cross_encoder(query, chunks_with_scores, top_k)
    #     
    #     # クロスエンコーダが使用できない場合は、そのまま返す
    #     if top_k is not None:
    #         return chunks_with_scores[:top_k]
    #     return chunks_with_scores
    
    # def _rerank_with_cross_encoder(
    #     self,
    #     query: str,
    #     chunks_with_scores: List[Tuple[Chunk, float]],
    #     top_k: Optional[int] = None
    # ) -> List[Tuple[Chunk, float]]:
    #     """
    #     クロスエンコーダを使用した再ランキング
    #     
    #     Args:
    #         query: クエリ
    #         chunks_with_scores: チャンクとスコアのタプルリスト
    #         top_k: 再ランキング後の取得数（Noneの場合は全て）
    #         
    #     Returns:
    #         再ランキング後のチャンクとスコアのリスト
    #     """
    #     if not chunks_with_scores:
    #         return []
    #     
    #     # クエリとチャンクのペアを作成
    #     pairs = [[query, chunk.content] for chunk, _ in chunks_with_scores]
    #     
    #     # クロスエンコーダでスコアを計算
    #     try:
    #         rerank_scores = self.cross_encoder.predict(pairs)
    #     except Exception as e:
    #         print(f"警告: クロスエンコーダの予測に失敗しました: {str(e)}")
    #         # エラーが発生した場合は元のスコアを使用
    #         if top_k is not None:
    #             return chunks_with_scores[:top_k]
    #         return chunks_with_scores
    #     
    #     # スコアでソート（降順）
    #     reranked_indices = np.argsort(rerank_scores)[::-1]
    #     
    #     # 再ランキング後の結果を作成
    #     reranked_results = [
    #         (chunks_with_scores[i][0], float(rerank_scores[i]))
    #         for i in reranked_indices
    #     ]
    #     
    #     # top_k で絞る
    #     if top_k is not None:
    #         return reranked_results[:top_k]
    #     return reranked_results
    
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

