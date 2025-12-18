"""
pipelineパッケージを直接実行する場合のエントリーポイント

使用方法:
    python -m RAG_pipeline.pipeline

または、PDFファイルを指定する場合:
    python -m RAG_pipeline.pipeline <pdf_path> [query]
"""

import sys
import os
from pathlib import Path

# 親ディレクトリをパスに追加
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import json
from RAG_pipeline import (
    AutoLoader,
    SemanticChunker,
    DummyEmbedder,
    Document,
    RAGPipeline
)


def main():
    """メイン関数"""
    print("=" * 80)
    print("RAGパイプライン実行")
    print("=" * 80)
    
    # コマンドライン引数からPDFパスを取得
    pdf_path = None
    query = None
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if len(sys.argv) > 2:
            query = sys.argv[2]
    
    # PDFパスが指定されていない場合、デフォルトのPDFファイルを探す
    if pdf_path is None:
        script_dir = Path(__file__).parent.parent
        default_pdf = script_dir / "mandel-freming-model.pdf"
        if default_pdf.exists():
            pdf_path = str(default_pdf)
        else:
            # PDFがない場合、テスト用のドキュメントを使用
            pdf_path = None
    
    try:
        # パイプラインの初期化
        loader = AutoLoader()
        embedder = DummyEmbedder(dimension=384)
        chunker = SemanticChunker(
            embedder = embedder,
            max_tokens = 700,
            min_tokens = 150,
        )
        
        pipeline = RAGPipeline(
            document_loader=loader,
            chunker=chunker,
            embedder=embedder
        )
        
        if pdf_path and os.path.exists(pdf_path):
            # PDFファイルを取り込む
            print(f"\nPDFファイルを読み込み中: {pdf_path}")
            result = pipeline.ingest(pdf_path)
            print("取り込み結果:", json.dumps(result, indent=2, ensure_ascii=False))
            
            if query is None:
                query = "マンデルフレミングモデルとは何ですか"
        else:
            # テスト用のドキュメントを作成
            test_doc = Document(
                content="これはテスト用のドキュメントです。RAGパイプラインの動作を確認するために使用されます。"
                        "機械学習と自然言語処理について説明します。"
                        "ベクトル検索と埋め込みについて学びます。",
                metadata={'source': 'test', 'category': 'example'}
            )
            
            result = pipeline.ingest_documents([test_doc])
            print("\n取り込み結果:", json.dumps(result, indent=2, ensure_ascii=False))
            
            if query is None:
                query = "ベクトル検索について"
        
        # 検索を実行
        print(f"\n検索クエリ: {query}")
        results = pipeline.search(query, top_k=3)
        
        print(f"検索結果数: {len(results)}")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n結果 {i} (スコア: {score:.4f}):")
            print(f"  内容: {chunk.content[:100]}...")
            print(f"  メタデータ: {chunk.metadata}")
        
        # 統計情報
        print("\nパイプライン統計:")
        print(json.dumps(pipeline.get_stats(), indent=2, ensure_ascii=False))
        
    except ImportError as e:
        print(f"エラー: 必要なライブラリがインストールされていません")
        print(f"詳細: {e}")
        print("\npypdfをインストールしてください: pip install pypdf")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

