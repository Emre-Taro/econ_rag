"""
RAGパイプラインモジュール

RAGPipelineを定義
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple

from ..models import Document, Chunk
from ..loading import DocumentLoader
from ..chunk import Chunker
from ..embedding import Embedder
from ..vectorstore import VectorStore
from ..retrievers import Retriever, SimpleRetriever
from ..enrich import enrich_document, enrich_chunk

# .envファイルの読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class RAGPipeline:
    """RAGパイプラインのメインクラス"""
    
    def __init__(
        self,
        document_loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        retriever: Optional[Retriever] = None,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        llm_temperature: float = 0.7
    ):
        """
        Args:
            document_loader: ドキュメント読み込み器
            chunker: チャンキング器
            embedder: 埋め込み生成器
            retriever: リトリーバー（Noneの場合は自動生成）
            llm_model: LLMモデル名（デフォルト: "gpt-4o-mini"）
            llm_api_key: OpenAI APIキー（Noneの場合は環境変数OPENAI_API_KEYを使用）
            llm_temperature: LLMの温度パラメータ（0.0-1.0）
        """
        self.document_loader = document_loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = VectorStore()
        
        if retriever is None:
            self.retriever = SimpleRetriever(self.vector_store, embedder)
        else:
            self.retriever = retriever
        
        # LLMの初期化
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self._llm_client = None
        
        # APIキーの取得
        if llm_api_key is None:
            llm_api_key = os.getenv("OPENAI_API_KEY")
        
        if llm_api_key:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=llm_api_key)
            except ImportError:
                print("警告: openaiライブラリがインストールされていません。LLM機能は使用できません。")
    
    def ingest(self, source: str, enrich_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ドキュメントを取り込む（インデックス化）
        
        Args:
            source: ドキュメントのソース
            enrich_context: metadata付与の合流点として注入する実行時context（任意）
            
        Returns:
            取り込み結果の統計情報
        """
        enrich_context = enrich_context or {}
        # ドキュメント読み込み
        documents = self.document_loader.load(source)
        
        all_chunks = []
        all_embeddings = []
        
        for doc in documents:
            # metadata付与の合流点（Document）
            enrich_document(doc, enrich_context)
            # チャンキング
            chunks = self.chunker.chunk(doc)

            # metadata付与の合流点（Chunk）
            for c in chunks:
                enrich_chunk(c, enrich_context)
            
            # 埋め込み生成
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedder.embed_batch(chunk_texts)
            
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
        
        # ベクトルストアに追加
        self.vector_store.add_chunks(all_chunks, all_embeddings)
        
        return {
            'documents_processed': len(documents),
            'chunks_created': len(all_chunks),
            'total_chunks_in_store': len(self.vector_store.chunks)
        }
    
    def ingest_documents(self, documents: List[Document], enrich_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        既存のDocumentオブジェクトを取り込む
        
        Args:
            documents: Documentのリスト
            enrich_context: metadata付与の合流点として注入する実行時context（任意）
            
        Returns:
            取り込み結果の統計情報
        """
        enrich_context = enrich_context or {}
        all_chunks = []
        all_embeddings = []
        
        for doc in documents:
            # metadata付与の合流点（Document）
            enrich_document(doc, enrich_context)
            chunks = self.chunker.chunk(doc)

            # metadata付与の合流点（Chunk）
            for c in chunks:
                enrich_chunk(c, enrich_context)

            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedder.embed_batch(chunk_texts)
            
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
        
        self.vector_store.add_chunks(all_chunks, all_embeddings)
        
        return {
            'documents_processed': len(documents),
            'chunks_created': len(all_chunks),
            'total_chunks_in_store': len(self.vector_store.chunks)
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        use_rerank: bool = False,
        rerank_top_k: Optional[int] = None,
        return_filter_stats: bool = False
    ) -> List[Tuple[Chunk, float]] | Tuple[List[Tuple[Chunk, float]], Dict[str, Any]]:
        """
        検索を実行
        
        Args:
            query: 検索クエリ
            top_k: 取得するチャンク数
            filters: フィルタ条件
            use_rerank: 再ランキングを使用するか
            rerank_top_k: 再ランキング後の取得数
            return_filter_stats: Trueの場合、フィルタリング統計情報も返す
            
        Returns:
            (Chunk, score)のタプルのリスト（return_filter_stats=Trueの場合は統計情報も含む）
        """
        from ..retrievers import SimpleRetriever
        
        # 検索
        if return_filter_stats and isinstance(self.retriever, SimpleRetriever):
            # SimpleRetrieverの場合、フィルタリング統計情報を取得
            results, filter_stats = self.retriever.retrieve(
                query, 
                top_k=top_k * 2 if use_rerank else top_k, 
                filters=filters,
                return_filter_stats=True
            )
        else:
            results = self.retriever.retrieve(query, top_k=top_k * 2 if use_rerank else top_k, filters=filters)
            filter_stats = {}
        
        # 再ランキング（拡張用）
        if use_rerank:
            results = self.retriever.rerank(query, results, top_k=rerank_top_k or top_k)
            if return_filter_stats:
                filter_stats["total_after_rerank"] = len(results)
        
        if return_filter_stats:
            return results, filter_stats
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """パイプラインの統計情報を取得"""
        return {
            'total_chunks': len(self.vector_store.chunks),
            'total_documents': len(set(chunk.doc_id for chunk in self.vector_store.chunks if chunk.doc_id)),
            'chunker_type': type(self.chunker).__name__,
            'embedder_type': type(self.embedder).__name__,
            'retriever_type': type(self.retriever).__name__,
            'llm_available': self._llm_client is not None,
            'llm_model': self.llm_model if self._llm_client else None
        }
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        return_filter_stats: bool = False
    ) -> Dict[str, Any]:
        """
        質問を受け取り、RAGを使ってLLMで回答を生成
        
        Args:
            question: ユーザーの質問
            top_k: 検索に使用するチャンク数
            system_prompt: システムプロンプト（Noneの場合はデフォルトを使用）
            max_tokens: 生成する最大トークン数
            return_filter_stats: Trueの場合、フィルタリング統計情報も返す
            
        Returns:
            回答と関連情報を含む辞書（return_filter_stats=Trueの場合は統計情報も含む）
        """
        if self._llm_client is None:
            raise ValueError(
                "LLMクライアントが初期化されていません。"
                "OpenAI APIキーを設定してください。"
            )
        
        # 1. 関連するチャンクを検索
        if return_filter_stats:
            search_results, filter_stats = self.search(question, top_k=top_k, return_filter_stats=True)
        else:
            search_results = self.search(question, top_k=top_k)
            filter_stats = {}
        
        if not search_results:
            result = {
                'answer': '関連する情報が見つかりませんでした。',
                'sources': [],
                'question': question
            }
            if return_filter_stats:
                result['filter_stats'] = filter_stats
            return result
        
        # 2. コンテキストを構築
        context_parts = []
        sources = []
        
        for i, (chunk, score) in enumerate(search_results, 1):
            context_parts.append(f"[文書{i}]\n{chunk.content}")
            
            # ソース情報を収集
            source_info = {
                'index': i,
                'score': score,
                'metadata': chunk.metadata.copy()
            }
            sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        
        # 3. プロンプトを構築
        if system_prompt is None:
            system_prompt = """あなたは与えられた文書を基に質問に答えるアシスタントです。
提供された文書の内容のみを基に回答してください。
文書に情報がない場合は、その旨を明確に伝えてください。
回答は日本語で行ってください。"""
        
        user_prompt = f"""以下の文書を参考にして、質問に答えてください。

【参考文書】
{context}

【質問】
{question}

【回答】"""
        
        # 4. LLMに質問を送信
        try:
            response = self._llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.llm_temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            
            result = {
                'answer': answer,
                'sources': sources,
                'question': question,
                'model': self.llm_model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            if return_filter_stats:
                result['filter_stats'] = filter_stats
            return result
        except Exception as e:
            raise RuntimeError(f"LLMからの回答生成中にエラーが発生しました: {str(e)}")
    
    def generate_answer(
        self,
        question: str,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000
    ) -> str:
        """
        質問を受け取り、回答のみを文字列で返す（シンプルなインターフェース）
        
        Args:
            question: ユーザーの質問
            top_k: 検索に使用するチャンク数
            system_prompt: システムプロンプト（Noneの場合はデフォルトを使用）
            max_tokens: 生成する最大トークン数
            
        Returns:
            生成された回答（文字列）
        """
        result = self.query(question, top_k=top_k, system_prompt=system_prompt, max_tokens=max_tokens)
        return result['answer']

