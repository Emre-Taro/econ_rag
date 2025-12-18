"""
PDFローダーのテストスクリプト
指定されたPDFファイルを読み込んで内容を表示します
"""

import sys
import os
from pathlib import Path

# 親ディレクトリをパスに追加（RAG_pipelineパッケージをインポートできるように）
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# RAG_pipelineパッケージから必要なクラスをインポート
from RAG_pipeline import PDFLoader, AutoLoader, Document

def test_pdf_loading(pdf_path: str):
    """
    PDFファイルを読み込んで内容を表示
    
    Args:
        pdf_path: PDFファイルのパス
    """
    # ファイルの存在確認
    if not os.path.exists(pdf_path):
        print(f"エラー: ファイルが見つかりません: {pdf_path}")
        return
    
    print(f"PDFファイルを読み込み中: {pdf_path}")
    print("=" * 80)
    
    try:
        # PDFローダーを使用
        loader = PDFLoader()
        documents = loader.load(pdf_path)
        
        print(f"\n読み込み完了: {len(documents)}ページのドキュメントを読み込みました\n")
        
        # 各ページの情報を表示
        for i, doc in enumerate(documents, 1):
            print(f"\n{'='*80}")
            print(f"ページ {i} / {len(documents)}")
            print(f"{'='*80}")
            
            # メタデータを表示
            print("\n【メタデータ】")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            
            # コンテンツを表示（最初の500文字）
            print(f"\n【コンテンツ（最初の500文字）】")
            content_preview = doc.content[:500]
            print(content_preview)
            if len(doc.content) > 500:
                print(f"\n... (残り {len(doc.content) - 500} 文字)")
            
            # 統計情報
            print(f"\n【統計】")
            print(f"  文字数: {len(doc.content)}")
            print(f"  単語数（概算）: {len(doc.content.split())}")
            
            # 最初の3ページだけ詳細表示、残りは簡潔に
            if i >= 3 and len(documents) > 3:
                print(f"\n... 残り {len(documents) - 3} ページをスキップします")
                print(f"\n全{len(documents)}ページのサマリー:")
                print(f"  総文字数: {sum(len(d.content) for d in documents)}")
                print(f"  平均文字数/ページ: {sum(len(d.content) for d in documents) // len(documents)}")
                break
        
        # 全体の統計
        print(f"\n{'='*80}")
        print("【全体統計】")
        print(f"{'='*80}")
        total_chars = sum(len(d.content) for d in documents)
        total_words = sum(len(d.content.split()) for d in documents)
        print(f"総ページ数: {len(documents)}")
        print(f"総文字数: {total_chars:,}")
        print(f"総単語数（概算）: {total_words:,}")
        print(f"平均文字数/ページ: {total_chars // len(documents):,}")
        
        # ファイルサイズ
        file_size = os.path.getsize(pdf_path)
        print(f"ファイルサイズ: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
        
    except ImportError as e:
        print(f"エラー: 必要なライブラリがインストールされていません")
        print(f"詳細: {e}")
        print("\npypdfをインストールしてください: pip install pypdf")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


def test_auto_loader(pdf_path: str):
    """
    AutoLoaderを使用してPDFを読み込むテスト
    
    Args:
        pdf_path: PDFファイルのパス
    """
    print(f"\n{'='*80}")
    print("AutoLoaderを使用した読み込みテスト")
    print(f"{'='*80}\n")
    
    try:
        loader = AutoLoader()
        documents = loader.load(pdf_path)
        
        print(f"AutoLoaderで読み込み成功: {len(documents)}ページ")
        print(f"最初のページのメタデータ: {documents[0].metadata if documents else 'なし'}")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # PDFファイルのパス
    pdf_path = r"C:\Users\emreh\OneDrive\Documents\Programming\AI\LLM_RAG\RAG\RAG_basic\RAG_pipeline\mandel-freming-model.pdf"
    
    # 相対パスも試す
    if not os.path.exists(pdf_path):
        # スクリプトと同じディレクトリを試す
        script_dir = Path(__file__).parent
        pdf_path = script_dir / "mandel-freming-model.pdf"
        pdf_path = str(pdf_path)
    
    print("PDFローダーテスト")
    print("=" * 80)
    
    # PDFLoaderで読み込み
    test_pdf_loading(pdf_path)
    
    # AutoLoaderで読み込み
    test_auto_loader(pdf_path)

