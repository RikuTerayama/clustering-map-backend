from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path


class AppConfig(BaseModel):
    """アプリケーション設定"""
    # データベース設定
    data_dir: str = Field("data", description="データディレクトリ")
    results_dir: str = Field("data/results", description="結果保存ディレクトリ")
    logs_dir: str = Field("logs", description="ログディレクトリ")
    temp_dir: str = Field("tmp", description="一時ファイルディレクトリ")
    
    # モデル設定
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="埋め込みモデル名"
    )
    
    # デフォルトパラメータ
    default_umap_params: Dict[str, Any] = Field(default_factory=lambda: {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "random_state": 42
    })
    
    default_hdbscan_params: Dict[str, Any] = Field(default_factory=lambda: {
        "min_cluster_size": 15,
        "min_samples": 5
    })
    
    default_kmeans_params: Dict[str, Any] = Field(default_factory=lambda: {
        "n_clusters": 8,
        "random_state": 42
    })
    
    # ファイル制限
    max_file_size: int = Field(50 * 1024 * 1024, description="最大ファイルサイズ（バイト）")
    max_rows: int = Field(50000, description="最大行数")
    
    # ログ設定
    log_level: str = Field("INFO", description="ログレベル")
    log_file: str = Field("logs/app.log", description="ログファイルパス")
    
    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> "AppConfig":
        """設定ファイルから読み込み"""
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            return cls(**config_data)
        return cls()
    
    def save_to_file(self, config_path: str = "config.json") -> None:
        """設定ファイルに保存"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2)
    
    def ensure_directories(self) -> None:
        """必要なディレクトリを作成"""
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir, self.temp_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
