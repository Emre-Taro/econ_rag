"""
埋め込み生成モジュール

Embedderとその実装クラスを定義
"""

from typing import List, Optional
import hashlib
import numpy as np
import os

# .envファイルの読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenvがインストールされていない場合は警告なしで続行
    pass


class Embedder:
    """埋め込み生成のベースクラス（拡張可能）"""
    
    def embed(self, text: str) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            text: 埋め込み対象のテキスト
            
        Returns:
            埋め込みベクトル（numpy配列）
        """
        raise NotImplementedError("サブクラスで実装してください")
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        複数のテキストを一括で埋め込み
        
        Args:
            texts: 埋め込み対象のテキストリスト
            
        Returns:
            埋め込みベクトルのリスト
        """
        return [self.embed(text) for text in texts]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        複数のテキストを一括で埋め込み（リスト形式で返す）
        SemanticChunkerで使用される形式
        
        Args:
            texts: 埋め込み対象のテキストリスト
            
        Returns:
            埋め込みベクトルのリスト（各ベクトルはfloatのリスト）
        """
        vectors = self.embed_batch(texts)
        return [v.tolist() for v in vectors]


class DummyEmbedder(Embedder):
    """ダミー埋め込み（テスト用）"""
    
    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: 埋め込みベクトルの次元数
        """
        self.dimension = dimension
    
    def embed(self, text: str) -> np.ndarray:
        """テキストのハッシュをベクトル化（ダミー実装）"""
        # 実際の実装では、sentence-transformers等を使用
        # ここではテスト用のダミー実装
        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest()[:16], 16)
        np.random.seed(hash_int % (2**32))
        vector = np.random.normal(0, 1, self.dimension)
        vector = vector / np.linalg.norm(vector)  # 正規化
        return vector


class OpenAIEmbedder(Embedder):
    """OpenAI Embeddings APIを使用した埋め込み生成"""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Args:
            model: OpenAIの埋め込みモデル名（例: "text-embedding-3-small", "text-embedding-ada-002"）
            api_key: OpenAI APIキー（Noneの場合は環境変数OPENAI_API_KEYを使用）
            dimension: 埋め込みベクトルの次元数（modelがtext-embedding-3-*の場合のみ指定可能）
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openaiライブラリがインストールされていません。"
                "以下のコマンドでインストールしてください: pip install openai"
            )
        
        self.model = model
        self.dimension = dimension
        
        # APIキーの取得
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI APIキーが指定されていません。"
                    "環境変数OPENAI_API_KEYを設定するか、api_key引数を指定してください。"
                )
        
        self.client = OpenAI(api_key=api_key)
    
    def embed(self, text: str) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            text: 埋め込み対象のテキスト
            
        Returns:
            埋め込みベクトル（numpy配列）
        """
        kwargs = {
            "model": self.model,
            "input": text
        }
        # dimensionsパラメータはtext-embedding-3-*モデルでのみ使用可能
        if self.dimension is not None and "text-embedding-3" in self.model:
            kwargs["dimensions"] = self.dimension
        
        response = self.client.embeddings.create(**kwargs)
        return np.array(response.data[0].embedding)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        複数のテキストを一括で埋め込み
        
        Args:
            texts: 埋め込み対象のテキストリスト
            
        Returns:
            埋め込みベクトルのリスト
        """
        # OpenAI APIは最大2048入力まで一度に処理可能
        max_batch_size = 2048
        
        # dimensionsパラメータはtext-embedding-3-*モデルでのみ使用可能
        kwargs = {"model": self.model}
        if self.dimension is not None and "text-embedding-3" in self.model:
            kwargs["dimensions"] = self.dimension
        
        if len(texts) <= max_batch_size:
            kwargs["input"] = texts
            response = self.client.embeddings.create(**kwargs)
            return [np.array(item.embedding) for item in response.data]
        else:
            # バッチサイズを超える場合は分割して処理
            vectors = []
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i:i + max_batch_size]
                batch_kwargs = kwargs.copy()
                batch_kwargs["input"] = batch
                response = self.client.embeddings.create(**batch_kwargs)
                vectors.extend([np.array(item.embedding) for item in response.data])
            return vectors

