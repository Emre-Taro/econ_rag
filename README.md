# RAG Pipeline 使用方法

## インストール

必要なライブラリをインストールしてください：

```bash
pip install -r requirement.txt
```

## 実行方法

### 方法1: パッケージとして実行（推奨）

親ディレクトリ（RAG_basic）から実行：

```bash
# パイプラインのデモを実行
python -m RAG_pipeline.pipeline

# または、example_usage.pyを実行
python -m RAG_pipeline.example_usage
```

または、RAG_pipelineディレクトリ内から実行：

```bash
cd RAG_pipeline
python example_usage.py
```

### 方法2: 別のスクリプトからインポート

```python
from RAG_pipeline import (
    AutoLoader,
    SemanticChunker,
    DummyEmbedder,
    Document,
    RAGPipeline
)

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

# ドキュメントを取り込む
result = pipeline.ingest("path/to/document.pdf")

# 検索を実行
results = pipeline.search("検索クエリ", top_k=5)
```

### 方法3: 個別のコンポーネントを使用

```python
from RAG_pipeline import PDFLoader, SemanticChunker, DummyEmbedder

# PDFを読み込む
loader = PDFLoader()
documents = loader.load("document.pdf")

# チャンキング
chunker = SemanticChunker(
    embedder = embedder,
    max_tokens = 700,
    min_tokens = 150,
)
chunks = chunker.chunk(documents[0])

# 埋め込み生成
embedder = DummyEmbedder(dimension=384)
embedding = embedder.embed("テキスト")
```

## テストの実行

```bash
# 親ディレクトリから実行
python -m RAG_pipeline.test_pdf_loader

# または、RAG_pipelineディレクトリ内から実行
cd RAG_pipeline
python test_pdf_loader.py
```

## ファイル構造

```
RAG_pipeline/
├── __init__.py          # パッケージのエクスポート
├── models.py            # Document, Chunk
├── loaders.py           # DocumentLoader, PDFLoader, AutoLoader等
├── chunkers.py          # Chunker, SemantciChunker 
├── embedders.py         # Embedder, DummyEmbedder
├── vector_store.py      # VectorStore
├── retrievers.py        # Retriever, SimpleRetriever
├── pipeline.py          # RAGPipeline
├── example_usage.py     # 使用例
└── test_pdf_loader.py   # PDFローダーのテスト
```

## 主なクラス

- **Document**: ドキュメントを表現するデータクラス
- **Chunk**: チャンクを表現するデータクラス
- **DocumentLoader**: ドキュメント読み込みのベースクラス
- **PDFLoader**: PDFファイルを読み込む
- **AutoLoader**: ファイル形式を自動判定して読み込む
- **Chunker**: チャンキングのベースクラス
- **SemanticChunker**: 意味チャンキング
- **Embedder**: 埋め込み生成のベースクラス
- **DummyEmbedder**: テスト用のダミー埋め込み
- **VectorStore**: ベクトルストア
- **Retriever**: 検索・リトリーバルのベースクラス
- **SimpleRetriever**: コサイン類似度ベースのリトリーバー
- **RAGPipeline**: 全体統合パイプライン

