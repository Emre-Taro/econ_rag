"""
RAGパイプラインの実行スクリプト

PDFファイルを読み込み、チャンキング、埋め込み、ベクトルストアへの保存、検索まで実行します。
"""

import sys
import os
import argparse
from pathlib import Path

# 親ディレクトリをパスに追加
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import json
from RAG_pipeline import (
    AutoLoader,
    SemanticChunker,
    OpenAIEmbedder,
    Document,
    RAGPipeline
)

def _build_enrich_context(args: argparse.Namespace) -> dict:
    """CLI引数から enrich 用の context を構築（metadata合流点に渡す）。"""
    ctx = {}
    for key in (
        "tenant_id",
        "index_version",
        "domain",
        "topic",
        "audience",
        "ingest_batch",
        "section",
        "content_type",
        "quality_score",
    ):
        val = getattr(args, key, None)
        if val is not None:
            ctx[key] = val
    return ctx


def run_rag_pipeline(
    pdf_path: str,
    query: str = None,
    enrich_context: dict | None = None,
):
    """
    RAGパイプラインを実行
    
    Args:
        pdf_path: PDFファイルのパス
        query: 検索クエリ（Noneの場合はデフォルトクエリを使用）
    """
    print("=" * 80)
    print("RAGパイプライン実行")
    print("=" * 80)
    
    # ファイルの存在確認
    if not os.path.exists(pdf_path):
        print(f"エラー: ファイルが見つかりません: {pdf_path}")
        return
    
    try:
        # 1. パイプラインコンポーネントの初期化
        print("\n[1/5] パイプラインコンポーネントの初期化...")
        loader = AutoLoader()
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        chunker = SemanticChunker(
            embedder = embedder,
            max_tokens = 700,
            min_tokens = 150,
        )
        
        pipeline = RAGPipeline(
            document_loader=loader,
            chunker=chunker,
            embedder=embedder,
            llm_model="gpt-4o-mini"
        )
        print("✓ 初期化完了")
        
        # 2. ドキュメントの取り込み（読み込み + チャンキング + 埋め込み + ベクトルストアに保存）
        print(f"\n[2/5] ドキュメントの取り込み: {pdf_path}")
        # 固定（デフォルト）metadata。引数で上書き可能。
        default_enrich_context = {
            "domain": "economics",
            "topic": "macro_model",
            "audience": "undergraduate",
        }
        merged_context = default_enrich_context
        if enrich_context:
            merged_context = {**default_enrich_context, **enrich_context}

        result = pipeline.ingest(pdf_path, enrich_context=merged_context)

        # 例: 全チャンクのmetadataをJSONLに保存
        out_path = "all_chunk_metadata.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for c in pipeline.vector_store.chunks:
                f.write(json.dumps(
                    {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "metadata": c.metadata},
                    ensure_ascii=False
                ) + "\n")
        print(f"\nSaved: {out_path}")
        # 例: 最初の5チャンクのmetadataを表示
        print("\n=== metadata sample (first 5 chunks) ===")
        for i, c in enumerate(pipeline.vector_store.chunks[:5], 1):
            print(f"\n--- chunk {i} ---")
            print(json.dumps(c.metadata, ensure_ascii=False, indent=2))


        print("✓ 取り込み完了")
        print(f"  - 処理されたドキュメント数: {result['documents_processed']}")
        print(f"  - 作成されたチャンク数: {result['chunks_created']}")
        print(f"  - ストア内の総チャンク数: {result['total_chunks_in_store']}")
        
        # 3. パイプラインの統計情報
        print("\n[3/5] パイプライン統計情報:")
        stats = pipeline.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        # 4. 質問に回答を生成
        if query is None:
            # デフォルトの質問
            query = "マンデルフレミングモデルとは何ですか？"
        
        print(f"\n[4/5] 質問: '{query}'")
        print("回答を生成中...")
        
        try:
            # LLMを使って回答を生成
            result = pipeline.query(query, top_k=5)
            answer = result['answer']
            
            print(f"\n[5/5] 生成された回答:")
            print("=" * 80)
            print(answer)
            print("=" * 80)
            
            # 使用したソースを表示
            if result.get('sources'):
                print(f"\n参照した文書数: {len(result['sources'])}")
                for source in result['sources'][:3]:  # 最初の3つを表示
                    if 'page_number' in source['metadata']:
                        print(f"  - ページ {source['metadata']['page_number']} (類似度: {source['score']:.4f})")
            
            # トークン使用量
            if 'usage' in result:
                print(f"\nトークン使用量: {result['usage']['total_tokens']} (プロンプト: {result['usage']['prompt_tokens']}, 生成: {result['usage']['completion_tokens']})")
            
            print("\n" + "=" * 80)
            print("RAGパイプライン実行完了")
            print("=" * 80)
            
            return pipeline, result
            
        except ValueError as e:
            print(f"\n警告: {e}")
            print("検索結果のみを表示します...")
            
            # LLMが使えない場合は検索結果のみ表示
            results = pipeline.search(query, top_k=5)
            print(f"✓ 検索完了: {len(results)}件の結果")
            
            print("\n検索結果:")
            print("=" * 80)
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\n結果 {i} (類似度スコア: {score:.4f})")
                print("-" * 80)
                
                if 'page_number' in chunk.metadata:
                    print(f"ページ: {chunk.metadata['page_number']}")
                if 'source' in chunk.metadata:
                    print(f"ソース: {os.path.basename(chunk.metadata['source'])}")
                
                print(f"\n内容:")
                content_preview = chunk.content[:300]
                print(content_preview)
                if len(chunk.content) > 300:
                    print(f"... (残り {len(chunk.content) - 300} 文字)")
            
            return pipeline, results
        
    except ImportError as e:
        print(f"\nエラー: 必要なライブラリがインストールされていません")
        print(f"詳細: {e}")
        print("\npypdfをインストールしてください: pip install pypdf")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode(pipeline: RAGPipeline):
    """
    インタラクティブな質問回答モード
    
    Args:
        pipeline: 初期化済みのRAGパイプライン
    """
    print("\n" + "=" * 80)
    print("インタラクティブ質問回答モード")
    print("質問を入力してください（終了するには 'exit' または 'quit' を入力）")
    print("=" * 80)
    
    while True:
        try:
            question = input("\n質問: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("終了します。")
                break
            
            if not question:
                continue
            
            try:
                # LLMを使って回答を生成
                print("回答を生成中...")
                result = pipeline.query(question, top_k=5, return_filter_stats=True)
                answer = result['answer']
                
                # フィルタリング情報を表示
                if result.get('filter_stats'):
                    filter_stats = result['filter_stats']
                    print("\n" + "=" * 80)
                    print("フィルタリング情報")
                    print("=" * 80)
                    print(f"フィルタ適用前の候補数: {filter_stats.get('total_before_filter', 'N/A')}")
                    print(f"フィルタ適用後の結果数: {filter_stats.get('total_after_filter', 'N/A')}")
                    print(f"\n適用されたフィルタ:")
                    for filter_name in filter_stats.get('active_filters', []):
                        print(f"  - {filter_name}")
                    
                    if filter_stats.get('inferred_content_type'):
                        print(f"\nクエリから推論されたcontent_type: {filter_stats['inferred_content_type']}")
                    
                    print(f"\nフィルタリング詳細:")
                    if filter_stats.get('filtered_by_quality_score', 0) > 0:
                        print(f"  - quality_scoreで除外: {filter_stats['filtered_by_quality_score']}件")
                    if filter_stats.get('filtered_by_avg_adj_similarity', 0) > 0:
                        print(f"  - avg_adj_similarityで除外: {filter_stats['filtered_by_avg_adj_similarity']}件")
                    if filter_stats.get('filtered_by_content_type', 0) > 0:
                        print(f"  - content_typeで除外: {filter_stats['filtered_by_content_type']}件")
                    if filter_stats.get('filtered_by_page_number', 0) > 0:
                        print(f"  - page_numberで除外: {filter_stats['filtered_by_page_number']}件")
                
                print(f"\n回答:")
                print("=" * 80)
                print(answer)
                print("=" * 80)
                
                # 使用したソースを表示
                if result.get('sources'):
                    print(f"\n参照した文書数: {len(result['sources'])}")
                    for source in result['sources'][:3]:
                        if 'page_number' in source['metadata']:
                            print(f"  - ページ {source['metadata']['page_number']} (類似度: {source['score']:.4f})")
                
            except ValueError as e:
                print(f"警告: {e}")
                print("検索結果のみを表示します...")
                
                # LLMが使えない場合は検索結果のみ表示
                results, filter_stats = pipeline.search(question, top_k=3, return_filter_stats=True)
                print(f"\n検索結果: {len(results)}件")
                
                # フィルタリング情報を表示
                print("\n" + "=" * 80)
                print("フィルタリング情報")
                print("=" * 80)
                print(f"フィルタ適用前の候補数: {filter_stats.get('total_before_filter', 'N/A')}")
                print(f"フィルタ適用後の結果数: {filter_stats.get('total_after_filter', 'N/A')}")
                print(f"\n適用されたフィルタ:")
                for filter_name in filter_stats.get('active_filters', []):
                    print(f"  - {filter_name}")
                
                if filter_stats.get('inferred_content_type'):
                    print(f"\nクエリから推論されたcontent_type: {filter_stats['inferred_content_type']}")
                
                print(f"\nフィルタリング詳細:")
                if filter_stats.get('filtered_by_quality_score', 0) > 0:
                    print(f"  - quality_scoreで除外: {filter_stats['filtered_by_quality_score']}件")
                if filter_stats.get('filtered_by_avg_adj_similarity', 0) > 0:
                    print(f"  - avg_adj_similarityで除外: {filter_stats['filtered_by_avg_adj_similarity']}件")
                if filter_stats.get('filtered_by_content_type', 0) > 0:
                    print(f"  - content_typeで除外: {filter_stats['filtered_by_content_type']}件")
                if filter_stats.get('filtered_by_page_number', 0) > 0:
                    print(f"  - page_numberで除外: {filter_stats['filtered_by_page_number']}件")
                
                print("\n" + "-" * 80)
                for i, (chunk, score) in enumerate(results, 1):
                    print(f"\n[{i}] スコア: {score:.4f}")
                    
                    # メタデータの詳細を表示
                    metadata = chunk.metadata or {}
                    if 'page_number' in metadata:
                        print(f"  ページ: {metadata['page_number']}")
                    if 'quality_score' in metadata:
                        print(f"  quality_score: {metadata['quality_score']}")
                    if 'avg_adj_similarity' in metadata:
                        print(f"  avg_adj_similarity: {metadata['avg_adj_similarity']}")
                    if 'content_type' in metadata:
                        print(f"  content_type: {metadata['content_type']}")
                    
                    print(f"  内容: {chunk.content[:200]}...")
                
        except KeyboardInterrupt:
            print("\n\n終了します。")
            break
        except Exception as e:
            print(f"エラー: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGパイプラインを実行して質問に回答を生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # PDFファイルと質問を指定
  python run_rag_pipeline.py document.pdf "マンデルフレミングモデルとは何ですか？"
  
  # デフォルトのPDFファイルを使用
  python run_rag_pipeline.py
  
  # インタラクティブモードで起動
  python run_rag_pipeline.py document.pdf --interactive
  
  # デフォルトPDFでインタラクティブモード
  python run_rag_pipeline.py --interactive
        """
    )
    
    parser.add_argument(
        'pdf_path',
        nargs='?',
        default=None,
        help='PDFファイルのパス（指定しない場合はデフォルトのPDFを使用）'
    )
    
    parser.add_argument(
        'question',
        nargs='?',
        default=None,
        help='質問（指定しない場合はデフォルトの質問を使用）'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='インタラクティブモードで起動（複数の質問が可能）'
    )
    
    parser.add_argument(
        '--llm-model',
        default='gpt-4o-mini',
        help='使用するLLMモデル（デフォルト: gpt-4o-mini）'
    )
    
    parser.add_argument(
        '--embedding-model',
        default='text-embedding-3-small',
        help='使用する埋め込みモデル（デフォルト: text-embedding-3-small）'
    )

    # enrich（metadata合流点）用の実行時注入パラメータ
    parser.add_argument('--tenant-id', dest='tenant_id', default=None, help='注入するtenant_id（任意）')
    parser.add_argument('--index-version', dest='index_version', default=None, help='注入するindex_version（任意）')
    # 固定（デフォルト）metadata：必要ならCLIで上書き可能
    parser.add_argument('--domain', dest='domain', default='economics', help='注入するdomain（デフォルト: economics）')
    parser.add_argument('--topic', dest='topic', default='macro_model', help='注入するtopic（デフォルト: macro_model）')
    parser.add_argument('--audience', dest='audience', default='undergraduate', help='注入するaudience（デフォルト: undergraduate）')
    parser.add_argument('--ingest-batch', dest='ingest_batch', default=None, help='注入するingest_batch（任意）')
    parser.add_argument('--section', dest='section', default=None, help='注入するsection（任意、Chunk向け）')
    parser.add_argument('--content-type', dest='content_type', default=None, help='注入するcontent_type（任意、Chunk向け）')
    parser.add_argument('--quality-score', dest='quality_score', type=float, default=None, help='注入するquality_score（任意、Chunk向け）')
    
    args = parser.parse_args()
    
    # PDFファイルのパスを決定
    if args.pdf_path:
        pdf_path = args.pdf_path
        # ファイルの存在確認
        if not os.path.exists(pdf_path):
            print(f"エラー: ファイルが見つかりません: {pdf_path}")
            print("\nヒント:")
            print("  - ファイルパスが正しいか確認してください")
            print("  - 相対パスの場合、現在のディレクトリからのパスを確認してください")
            script_dir = Path(__file__).parent
            default_pdf = script_dir / "mandel-freming-model.pdf"
            if default_pdf.exists():
                print(f"  - デフォルトのPDFファイルが見つかりました: {default_pdf}")
                print(f"    使用する場合は: python run_rag_pipeline.py")
            sys.exit(1)
    else:
        # デフォルトのPDFファイルを探す
        script_dir = Path(__file__).parent
        default_pdf = script_dir / "mandel-freming-model.pdf"
        if default_pdf.exists():
            pdf_path = str(default_pdf.resolve())
            print(f"デフォルトのPDFファイルを使用: {pdf_path}")
        else:
            print("エラー: PDFファイルが指定されておらず、デフォルトのPDFファイルも見つかりません。")
            print("\n使用方法:")
            print("  python run_rag_pipeline.py <pdf_path> [question]")
            print("  または")
            print("  python run_rag_pipeline.py <pdf_path> --interactive")
            print("\n例:")
            print("  python run_rag_pipeline.py document.pdf \"質問内容\"")
            sys.exit(1)
    
    # 質問を決定
    question = args.question

    enrich_context = _build_enrich_context(args)
    
    # RAGパイプラインを実行
    try:
        # パイプラインコンポーネントの初期化
        print("=" * 80)
        print("RAGパイプライン実行")
        print("=" * 80)
        print(f"\n[1/3] パイプラインコンポーネントの初期化...")
        loader = AutoLoader()
        embedder = OpenAIEmbedder(model=args.embedding_model)
        chunker = SemanticChunker(
            embedder=embedder,
            max_tokens=700,
            min_tokens=150,
        )
        
        pipeline = RAGPipeline(
            document_loader=loader,
            chunker=chunker,
            embedder=embedder,
            llm_model=args.llm_model
        )
        print("✓ 初期化完了")
        
        # ドキュメントの取り込み
        print(f"\n[2/3] ドキュメントの取り込み: {pdf_path}")
        result = pipeline.ingest(pdf_path, enrich_context=enrich_context)
        # 例: 全チャンクのmetadataをJSONLに保存
        out_path = "all_chunk_metadata.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for c in pipeline.vector_store.chunks:
                f.write(json.dumps(
                    {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "metadata": c.metadata},
                    ensure_ascii=False
                ) + "\n")
        print(f"\nSaved: {out_path}")

        print("✓ 取り込み完了")
        print(f"  - 処理されたドキュメント数: {result['documents_processed']}")
        print(f"  - 作成されたチャンク数: {result['chunks_created']}")
        print(f"  - ストア内の総チャンク数: {result['total_chunks_in_store']}")
        
        # インタラクティブモードの場合
        if args.interactive:
            interactive_mode(pipeline)
        else:
            # 質問に回答を生成
            if question is None:
                question = "マンデルフレミングモデルとは何ですか？"
            
            print(f"\n[3/3] 質問: '{question}'")
            print("回答を生成中...")
            
            try:
                result = pipeline.query(question, top_k=5, return_filter_stats=True)
                answer = result['answer']
                
                # フィルタリング情報を表示
                if result.get('filter_stats'):
                    filter_stats = result['filter_stats']
                    print("\n" + "=" * 80)
                    print("フィルタリング情報")
                    print("=" * 80)
                    print(f"フィルタ適用前の候補数: {filter_stats.get('total_before_filter', 'N/A')}")
                    print(f"フィルタ適用後の結果数: {filter_stats.get('total_after_filter', 'N/A')}")
                    print(f"\n適用されたフィルタ:")
                    for filter_name in filter_stats.get('active_filters', []):
                        print(f"  - {filter_name}")
                    
                    if filter_stats.get('inferred_content_type'):
                        print(f"\nクエリから推論されたcontent_type: {filter_stats['inferred_content_type']}")
                    
                    print(f"\nフィルタリング詳細:")
                    if filter_stats.get('filtered_by_quality_score', 0) > 0:
                        print(f"  - quality_scoreで除外: {filter_stats['filtered_by_quality_score']}件")
                    if filter_stats.get('filtered_by_avg_adj_similarity', 0) > 0:
                        print(f"  - avg_adj_similarityで除外: {filter_stats['filtered_by_avg_adj_similarity']}件")
                    if filter_stats.get('filtered_by_content_type', 0) > 0:
                        print(f"  - content_typeで除外: {filter_stats['filtered_by_content_type']}件")
                    if filter_stats.get('filtered_by_page_number', 0) > 0:
                        print(f"  - page_numberで除外: {filter_stats['filtered_by_page_number']}件")
                    
                    # 使用したソースのメタデータを表示
                    if result.get('sources'):
                        print(f"\n使用したソースのメタデータ:")
                        for source in result['sources'][:3]:
                            metadata = source.get('metadata', {})
                            print(f"\n  ソース {source.get('index', 'N/A')} (類似度: {source.get('score', 0):.4f}):")
                            if 'page_number' in metadata:
                                print(f"    - ページ番号: {metadata['page_number']}")
                            if 'quality_score' in metadata:
                                print(f"    - quality_score: {metadata['quality_score']}")
                            if 'avg_adj_similarity' in metadata:
                                print(f"    - avg_adj_similarity: {metadata['avg_adj_similarity']}")
                            if 'content_type' in metadata:
                                print(f"    - content_type: {metadata['content_type']}")
                
                print(f"\n生成された回答:")
                print("=" * 80)
                print(answer)
                print("=" * 80)
                
                # 使用したソースを表示
                if result.get('sources'):
                    print(f"\n参照した文書数: {len(result['sources'])}")
                    for source in result['sources'][:3]:
                        if 'page_number' in source['metadata']:
                            print(f"  - ページ {source['metadata']['page_number']} (類似度: {source['score']:.4f})")
                
                # トークン使用量
                if 'usage' in result:
                    print(f"\nトークン使用量: {result['usage']['total_tokens']} (プロンプト: {result['usage']['prompt_tokens']}, 生成: {result['usage']['completion_tokens']})")
                
                print("\n" + "=" * 80)
                print("RAGパイプライン実行完了")
                print("=" * 80)
                
            except ValueError as e:
                print(f"\n警告: {e}")
                print("検索結果のみを表示します...")
                
                results, filter_stats = pipeline.search(question, top_k=5, return_filter_stats=True)
                print(f"✓ 検索完了: {len(results)}件の結果")
                
                # フィルタリング情報を表示
                print("\n" + "=" * 80)
                print("フィルタリング情報")
                print("=" * 80)
                print(f"フィルタ適用前の候補数: {filter_stats.get('total_before_filter', 'N/A')}")
                print(f"フィルタ適用後の結果数: {filter_stats.get('total_after_filter', 'N/A')}")
                print(f"\n適用されたフィルタ:")
                for filter_name in filter_stats.get('active_filters', []):
                    print(f"  - {filter_name}")
                
                if filter_stats.get('inferred_content_type'):
                    print(f"\nクエリから推論されたcontent_type: {filter_stats['inferred_content_type']}")
                
                print(f"\nフィルタリング詳細:")
                if filter_stats.get('filtered_by_quality_score', 0) > 0:
                    print(f"  - quality_scoreで除外: {filter_stats['filtered_by_quality_score']}件")
                if filter_stats.get('filtered_by_avg_adj_similarity', 0) > 0:
                    print(f"  - avg_adj_similarityで除外: {filter_stats['filtered_by_avg_adj_similarity']}件")
                if filter_stats.get('filtered_by_content_type', 0) > 0:
                    print(f"  - content_typeで除外: {filter_stats['filtered_by_content_type']}件")
                if filter_stats.get('filtered_by_page_number', 0) > 0:
                    print(f"  - page_numberで除外: {filter_stats['filtered_by_page_number']}件")
                
                print("\n検索結果:")
                print("=" * 80)
                for i, (chunk, score) in enumerate(results, 1):
                    print(f"\n結果 {i} (類似度スコア: {score:.4f})")
                    print("-" * 80)
                    
                    # メタデータの詳細を表示
                    metadata = chunk.metadata or {}
                    print("メタデータ:")
                    if 'page_number' in metadata:
                        print(f"  - ページ番号: {metadata['page_number']}")
                    if 'source' in metadata:
                        print(f"  - ソース: {os.path.basename(metadata['source'])}")
                    if 'quality_score' in metadata:
                        print(f"  - quality_score: {metadata['quality_score']}")
                    if 'avg_adj_similarity' in metadata:
                        print(f"  - avg_adj_similarity: {metadata['avg_adj_similarity']}")
                    if 'content_type' in metadata:
                        print(f"  - content_type: {metadata['content_type']}")
                    if 'chunk_index' in metadata:
                        print(f"  - chunk_index: {metadata['chunk_index']}")
                    
                    print(f"\n内容:")
                    content_preview = chunk.content[:300]
                    print(content_preview)
                    if len(chunk.content) > 300:
                        print(f"... (残り {len(chunk.content) - 300} 文字)")
    
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

