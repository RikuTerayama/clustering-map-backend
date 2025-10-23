FROM python:3.11-slim

# システム依存関係のインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# Python依存関係のインストール（Rustコンパイルを回避）
COPY requirements-ultra-minimal.txt ./
RUN pip install --no-cache-dir --no-build-isolation -r requirements-ultra-minimal.txt

# アプリケーションコードのコピー
COPY . ./

# ポートの公開
EXPOSE 8000

# ヘルスチェック用のエンドポイントを追加
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# アプリケーションの起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
