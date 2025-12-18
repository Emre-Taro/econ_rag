"""
SemanticChunkerの動作を確認するスクリプト

PDFファイルを読み込んで、SemanticChunkerでチャンキングした結果を表示します。
"""

import sys
import os
from pathlib import Path
import json

# Windowsでのエンコーディング問題を回避
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 親ディレクトリをパスに追加（RAG_pipelineパッケージをインポートできるように）
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from RAG_pipeline import (
    PDFLoader,
    SemanticChunker,
    OpenAIEmbedder,
    Document
)


def count_tokens_simple(text: str) -> int:
    """
    シンプルなトークンカウンタ（文字数ベースの概算）
    実際の使用では、tiktoken等を使用することを推奨
    """
    # 日本語と英語を考慮した簡易的なカウント
    # 日本語は約2文字=1トークン、英語は約4文字=1トークンとして概算
    japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FAF')
    other_chars = len(text) - japanese_chars
    return int(japanese_chars / 2 + other_chars / 4)


def test_semantic_chunking_full_document(pdf_path: str = None):
    """
    PDF全体を1つのDocumentとして結合してからSemanticChunkerでチャンキング
    
    Args:
        pdf_path: PDFファイルのパス（Noneの場合はデフォルトのPDFを使用）
    """
    print("=" * 80)
    print("SemanticChunker テスト（PDF全体を1つのDocumentとして処理）")
    print("=" * 80)
    
    # PDFファイルのパスを決定
    if pdf_path is None:
        script_dir = Path(__file__).parent
        pdf_path = script_dir / "mandel-freming-model.pdf"
    
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"\nエラー: PDFファイルが見つかりません: {pdf_path}")
        print("PDFファイルのパスを指定してください。")
        return
    
    try:
        # 1. PDFを読み込む
        print(f"\n[1/4] PDFファイルを読み込み中: {pdf_path}")
        loader = PDFLoader()
        documents = loader.load(str(pdf_path))
        print(f"[OK] 読み込み完了: {len(documents)}ページ")
        
        # 全ページのテキストを結合
        print("\n[2/4] 全ページを1つのDocumentに結合中...")
        page_texts = [doc.content for doc in documents]
        full_text = "\n\n".join(page_texts)
        
        # メタデータを結合（最初のページのメタデータをベースに）
        combined_metadata = documents[0].metadata.copy() if documents else {}
        combined_metadata.update({
            "total_pages": len(documents),
            "combined": True,
            "source_file": str(pdf_path)
        })
        
        # 1つのDocumentオブジェクトを作成
        doc = Document(
            content=full_text,
            metadata=combined_metadata,
            doc_id="mf_full"
        )
        
        total_chars = len(full_text)
        print(f"[OK] 結合完了")
        print(f"  総文字数: {total_chars:,}")
        print(f"  総ページ数: {len(documents)}")
        
        # 3. EmbedderとChunkerを初期化
        print("\n[3/4] EmbedderとChunkerを初期化中...")
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        chunker = SemanticChunker(
            embedder=embedder,
            max_tokens=700,
            min_tokens=150,
            similarity_threshold=0.72,
            hard_break_ratio=0.15,
            token_counter=count_tokens_simple
        )
        print("[OK] 初期化完了")
        
        # 4. チャンキングを実行
        print("\n[4/4] チャンキングを実行中...")
        chunks = chunker.chunk(doc)
        print(f"[OK] チャンキング完了")
        print(f"\n総チャンク数: {len(chunks)}")
        
        # 最初の5つのチャンクの詳細を表示（全文表示）
        print("\n" + "=" * 80)
        print("チャンク詳細（最初の5つ - 全文表示）")
        print("=" * 80)
        for i, chunk in enumerate(chunks[:5]):
            tokens = count_tokens_simple(chunk.content)
            reason = chunk.metadata.get('split_reason', 'unknown')
            avg_sim = chunk.metadata.get('avg_adj_similarity', 'N/A')
            
            print(f"\n{'=' * 80}")
            print(f"チャンク {i + 1}:")
            print(f"{'=' * 80}")
            print(f"トークン数: {tokens}")
            print(f"文字数: {len(chunk.content):,}")
            print(f"分割理由: {reason}")
            if isinstance(avg_sim, (int, float)):
                print(f"平均隣接類似度: {avg_sim:.3f}")
            print(f"\n--- 全文 ---")
            print(chunk.content)
            print(f"\n{'=' * 80}")
        
        # 統計情報
        print("\n" + "=" * 80)
        print("統計情報")
        print("=" * 80)
        
        chunk_tokens = [count_tokens_simple(c.content) for c in chunks]
        chunk_chars = [len(c.content) for c in chunks]
        
        print(f"総チャンク数: {len(chunks)}")
        print(f"平均トークン数: {sum(chunk_tokens) / len(chunk_tokens):.1f}")
        print(f"最小トークン数: {min(chunk_tokens)}")
        print(f"最大トークン数: {max(chunk_tokens)}")
        print(f"平均文字数: {sum(chunk_chars) / len(chunk_chars):.1f}")
        
        # 分割理由の分布
        split_reasons = {}
        for chunk in chunks:
            reason = chunk.metadata.get('split_reason', 'unknown')
            split_reasons[reason] = split_reasons.get(reason, 0) + 1
        
        print("\n分割理由の分布:")
        for reason, count in sorted(split_reasons.items()):
            print(f"  {reason}: {count} ({count/len(chunks)*100:.1f}%)")
        
        # 詳細な結果をJSONファイルに保存
        output_file = Path(__file__).parent / "chunking_results_full_document.json"
        results = {
            "pdf_path": str(pdf_path),
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "processing_mode": "full_document",
            "chunks": [
                {
                    "chunk_index": i,
                    "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                    "tokens": count_tokens_simple(chunk.content),
                    "chars": len(chunk.content),
                    "metadata": chunk.metadata
                }
                for i, chunk in enumerate(chunks)
            ],
            "statistics": {
                "avg_tokens": sum(chunk_tokens) / len(chunk_tokens),
                "min_tokens": min(chunk_tokens),
                "max_tokens": max(chunk_tokens),
                "split_reasons": split_reasons
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n詳細な結果を保存しました: {output_file}")
        
    except ImportError as e:
        print(f"\nエラー: 必要なライブラリがインストールされていません")
        print(f"詳細: {e}")
        print("\npypdfをインストールしてください: pip install pypdf")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SemanticChunkerのテスト（PDF全体を1つのDocumentとして処理）")
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="PDFファイルのパス（指定しない場合はデフォルトのPDFを使用）"
    )
    
    args = parser.parse_args()
    
    test_semantic_chunking_full_document(args.pdf)

