"""
ドキュメント読み込みモジュール

DocumentLoaderとその実装クラスを定義
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models import Document


def normalize_text(text: str) -> str:
    """
    テキストを正規化する（LLMが読みやすい形式に変換）
    
    Args:
        text: 正規化対象のテキスト
        
    Returns:
        正規化されたテキスト
    """
    # 「↓」を「よって」に変換
    normalized = text.replace('↓', 'よって')
    return normalized


class DocumentLoader:
    """ドキュメント読み込みのベースクラス（拡張可能）"""
    
    def load(self, source: str) -> List[Document]:
        """
        ドキュメントを読み込む
        
        Args:
            source: ドキュメントのソース（ファイルパス、URLなど）
            
        Returns:
            Documentのリスト
        """
        raise NotImplementedError("サブクラスで実装してください")
    
    def load_from_text(self, text: str, metadata: Optional[Dict] = None) -> Document:
        """
        テキストから直接Documentを作成
        
        Args:
            text: テキスト内容
            metadata: メタデータ
            
        Returns:
            Documentオブジェクト
        """
        if metadata is None:
            metadata = {}
        return Document(content=text, metadata=metadata)


class SimpleTextLoader(DocumentLoader):
    """シンプルなテキストファイル読み込み"""
    
    def load(self, source: str) -> List[Document]:
        """テキストファイルを読み込む"""
        with open(source, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # テキスト正規化（↓をよってに変換など）
        content = normalize_text(content)
        
        metadata = {
            'source': source,
            'file_name': os.path.basename(source),
            'file_type': 'text',
            'loaded_at': datetime.now().isoformat()
        }
        
        return [Document(content=content, metadata=metadata)]


class PDFLoader(DocumentLoader):
    """PDFファイル読み込み"""
    
    def __init__(self):
        """PDFローダーの初期化"""
        try:
            import pypdf
            self.pypdf = pypdf
        except ImportError:
            raise ImportError(
                "pypdfライブラリが必要です。インストールしてください: pip install pypdf"
            )
    
    def load(self, source: str) -> List[Document]:
        """
        PDFファイルを読み込む
        
        Args:
            source: PDFファイルのパス
            
        Returns:
            Documentのリスト（各ページが1つのDocument）
        """
        documents = []
        
        with open(source, 'rb') as file:
            pdf_reader = self.pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                content = page.extract_text()
                
                # テキスト正規化（↓をよってに変換など）
                content = normalize_text(content)
                
                # 空のページはスキップ
                if not content.strip():
                    continue
                
                metadata = {
                    'source': source,
                    'file_name': os.path.basename(source),
                    'file_type': 'pdf',
                    'page_number': page_num + 1,
                    'total_pages': total_pages,
                    'loaded_at': datetime.now().isoformat()
                }
                
                # PDFのメタデータがあれば追加
                if pdf_reader.metadata:
                    if pdf_reader.metadata.title:
                        metadata['pdf_title'] = pdf_reader.metadata.title
                    if pdf_reader.metadata.author:
                        metadata['pdf_author'] = pdf_reader.metadata.author
                
                documents.append(Document(content=content, metadata=metadata))
        
        return documents


class AutoLoader(DocumentLoader):
    """ファイル形式を自動判定して読み込むローダー"""
    
    def __init__(self):
        """自動ローダーの初期化"""
        self.text_loader = SimpleTextLoader()
        self.pdf_loader = None  # 必要になったら初期化
    
    def _get_file_extension(self, source: str) -> str:
        """ファイル拡張子を取得"""
        return os.path.splitext(source)[1].lower()
    
    def load(self, source: str) -> List[Document]:
        """
        ファイル形式を自動判定して読み込む
        
        Args:
            source: ファイルパス
            
        Returns:
            Documentのリスト
        """
        ext = self._get_file_extension(source)
        
        if ext == '.pdf':
            if self.pdf_loader is None:
                self.pdf_loader = PDFLoader()
            return self.pdf_loader.load(source)
        elif ext in ['.txt', '.md', '.text']:
            return self.text_loader.load(source)
        else:
            # デフォルトはテキストとして扱う
            try:
                return self.text_loader.load(source)
            except Exception as e:
                raise ValueError(
                    f"サポートされていないファイル形式です: {ext}. "
                    f"エラー: {str(e)}"
                )

