"""
検索・リトリーバルモジュール

Retrieverとその実装クラスを定義
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import re
from collections import Counter

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
        initial_retrieval_top_n: int = 15,
        use_query_expansion: bool = True,  # クエリ拡張を有効化
        expansion_top_k: int = 3,  # 拡張に使用する初期検索結果数
        expansion_num_terms: int = 5,  # 拡張に追加する語句数
        expansion_weight: float = 0.5  # 拡張クエリの重み（0.0-1.0）
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
            use_query_expansion: クエリ拡張を使用するか（デフォルト: True）
            expansion_top_k: 拡張に使用する初期検索結果数（デフォルト: 3）
            expansion_num_terms: 拡張に追加する語句数（デフォルト: 5）
            expansion_weight: 拡張クエリの重み（0.0-1.0、デフォルト: 0.5）
        """
        super().__init__(vector_store, embedder)
        self.min_quality_score = min_quality_score
        self.min_avg_adj_similarity = min_avg_adj_similarity
        self.excluded_page_numbers = excluded_page_numbers if excluded_page_numbers is not None else [7, 10]
        self.use_rerank = use_rerank
        self.rerank_top_k = rerank_top_k
        self.initial_retrieval_top_n = initial_retrieval_top_n
        self.use_query_expansion = use_query_expansion
        self.expansion_top_k = expansion_top_k
        self.expansion_num_terms = expansion_num_terms
        self.expansion_weight = expansion_weight
        
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
        
        # クエリ拡張を適用
        if self.use_query_expansion and len(results) > 0:
            expanded_results = self._apply_query_expansion(query, results, top_k)
            if return_filter_stats:
                filter_stats["query_expansion_applied"] = True
                filter_stats["expansion_terms"] = self._extract_expansion_terms(
                    results[:self.expansion_top_k], 
                    original_query=query
                )
            results = expanded_results
        
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
        
        # 最終結果を top_k で取得
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
    
    def _extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """
        テキストからキーワードを抽出
        
        Args:
            text: 抽出対象のテキスト
            min_length: 最小文字数
            
        Returns:
            キーワードのリスト
        """
        # 日本語と英語の単語を抽出
        # 日本語: ひらがな、カタカナ、漢字の連続
        # 英語: アルファベットの連続
        japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+'
        english_pattern = r'[a-zA-Z]+'
        
        keywords = []
        # 日本語の単語を抽出
        japanese_words = re.findall(japanese_pattern, text)
        keywords.extend([w for w in japanese_words if len(w) >= min_length])
        
        # 英語の単語を抽出
        english_words = re.findall(english_pattern, text)
        keywords.extend([w.lower() for w in english_words if len(w) >= min_length])
        
        return keywords
    
    def _extract_expansion_terms(
        self, 
        top_results: List[Tuple[Chunk, float]],
        original_query: Optional[str] = None
    ) -> List[str]:
        """
        検索結果から拡張用の語句を抽出
        
        Args:
            top_results: 上位の検索結果
            original_query: 元のクエリ（キーワード除外用）
            
        Returns:
            拡張用の語句リスト
        """
        if not top_results:
            return []
        
        # 全てのチャンクからキーワードを抽出
        all_keywords = []
        for chunk, score in top_results:
            keywords = self._extract_keywords(chunk.content)
            all_keywords.extend(keywords)
        
        # 頻度をカウント
        keyword_counts = Counter(all_keywords)
        
        # 元のクエリのキーワードを除外
        query_keywords = set()
        if original_query:
            query_keywords = set(self._extract_keywords(original_query))
        
        # 頻度の高いキーワードを選択（元のクエリに含まれないもの）
        expansion_terms = [
            term for term, count in keyword_counts.most_common(self.expansion_num_terms * 2)
            if term not in query_keywords
        ]
        
        return expansion_terms[:self.expansion_num_terms]
    
    def _apply_query_expansion(
        self,
        original_query: str,
        initial_results: List[Tuple[Chunk, float]],
        top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """
        クエリ拡張を適用して検索結果を改善
        
        Args:
            original_query: 元のクエリ
            initial_results: 初期検索結果
            top_k: 最終的に取得するチャンク数
            
        Returns:
            拡張後の検索結果
        """
        if not initial_results or len(initial_results) < self.expansion_top_k:
            return initial_results
        
        # 上位の検索結果から拡張語句を抽出
        top_results = initial_results[:self.expansion_top_k]
        expansion_terms = self._extract_expansion_terms(top_results, original_query=original_query)
        
        if not expansion_terms:
            return initial_results
        
        # 拡張クエリを作成
        expanded_query = f"{original_query} {' '.join(expansion_terms)}"
        
        # クエリ拡張の情報を出力
        print("\n" + "=" * 80)
        print("[クエリ拡張]")
        print(f"元のクエリ: {original_query}")
        print(f"抽出された拡張語句: {', '.join(expansion_terms)}")
        print(f"拡張後のクエリ: {expanded_query}")
        print("=" * 80 + "\n")
        
        # 拡張クエリで再検索（フィルタリングなしで高速に検索）
        expanded_results = self._search_with_query(expanded_query, top_k * 2)
        
        # 元のクエリと拡張クエリの結果をマージ
        merged_results = self._merge_search_results(
            original_query,
            initial_results,
            expanded_query,
            expanded_results,
            top_k
        )
        
        return merged_results
    
    def _search_with_query(
        self,
        query: str,
        top_k: int,
        apply_filters: bool = False
    ) -> List[Tuple[Chunk, float]]:
        """
        クエリで検索を実行（フィルタリングなしの高速版）
        
        Args:
            query: 検索クエリ
            top_k: 取得するチャンク数
            apply_filters: フィルタリングを適用するか
            
        Returns:
            検索結果
        """
        if len(self.vector_store.chunks) == 0:
            return []
        
        # クエリを埋め込み
        query_embedding = self.embedder.embed(query)
        
        # コサイン類似度を計算
        scores = []
        for embedding in self.vector_store.embeddings:
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
        
        # フィルタリングを適用（オプション）
        if apply_filters:
            results, _ = self._apply_default_filters(query, results)
        
        return results[:top_k]
    
    def _merge_search_results(
        self,
        original_query: str,
        original_results: List[Tuple[Chunk, float]],
        expanded_query: str,
        expanded_results: List[Tuple[Chunk, float]],
        top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """
        元のクエリと拡張クエリの検索結果をマージ
        
        Args:
            original_query: 元のクエリ
            original_results: 元のクエリの検索結果
            expanded_query: 拡張クエリ
            expanded_results: 拡張クエリの検索結果
            top_k: 最終的に取得するチャンク数
            
        Returns:
            マージ後の検索結果
        """
        # チャンクIDをキーとして結果を統合
        merged_scores: Dict[str, Tuple[Chunk, float, float]] = {}
        
        # 元のクエリの結果を追加（重み: 1.0）
        for chunk, score in original_results:
            chunk_id = chunk.chunk_id if chunk.chunk_id else chunk.content[:50]  # チャンクIDまたは内容の一部
            merged_scores[chunk_id] = (chunk, score, 1.0)
        
        # 拡張クエリの結果を追加（重み: expansion_weight）
        for chunk, score in expanded_results:
            chunk_id = chunk.chunk_id if chunk.chunk_id else chunk.content[:50]
            if chunk_id in merged_scores:
                # 既に存在する場合は、重み付き平均でスコアを更新
                orig_chunk, orig_score, orig_weight = merged_scores[chunk_id]
                new_weight = orig_weight + self.expansion_weight
                new_score = (orig_score * orig_weight + score * self.expansion_weight) / new_weight
                merged_scores[chunk_id] = (chunk, new_score, new_weight)
            else:
                # 新規の場合は拡張クエリの重みで追加
                merged_scores[chunk_id] = (chunk, score, self.expansion_weight)
        
        # スコアでソート
        merged_list = [
            (chunk, score) for chunk, score, _ in merged_scores.values()
        ]
        merged_list.sort(key=lambda x: x[1], reverse=True)
        
        return merged_list[:top_k]

