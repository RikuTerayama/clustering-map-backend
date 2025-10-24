import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path

from app.models.schemas import (
    AnalysisRequest, DataPoint, TagRule, ColumnMapping
)
from app.models.config import AppConfig
from app.services.excel_service import ExcelService
from app.utils.text_utils import preprocess_text

logger = logging.getLogger(__name__)


class AnalysisService:
    """データ解析サービス"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.excel_service = ExcelService(config)
        self.sentence_model = None
        self.current_data = None
        self.current_config = None
    
    def _get_sentence_model(self):
        """SentenceTransformerモデルを取得（遅延読み込み）"""
        if self.sentence_model is None:
            try:
                self.sentence_model = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                logger.error(f"Failed to load sentence model: {e}")
                raise
        return self.sentence_model
    
    def analyze_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """データの解析を実行"""
        try:
            # 現在の設定を保存
            self.current_config = request.model_dump()
            
            # データを読み込み（実際の実装では、アップロードされたデータを取得）
            # ここでは仮のデータを使用
            df = self._load_current_data()
            if df is None:
                raise ValueError("解析するデータが見つかりません。先にファイルをアップロードしてください。")
            
            # データの前処理
            processed_df = self.excel_service.preprocess_data(df, request.column_mapping)
            
            # テキストの埋め込みベクトル化
            logger.info("Generating embeddings...")
            embeddings = self._generate_embeddings(
                processed_df[request.column_mapping.text_column].tolist()
            )
            
            # UMAP次元圧縮
            logger.info("Applying UMAP...")
            umap_coords = self._apply_umap(embeddings, request.umap_params)
            
            # クラスタリング
            logger.info("Performing clustering...")
            cluster_labels = self._perform_clustering(
                embeddings, request.cluster_method, 
                request.hdbscan_params, request.kmeans_params
            )
            
            # タグ生成と適用
            logger.info("Generating tags...")
            tags = self._generate_and_apply_tags(
                processed_df[request.column_mapping.text_column].tolist(),
                request.tag_rules
            )
            
            # 図形マスクへのスナップ（オプション）
            if request.shape_mask_path and os.path.exists(request.shape_mask_path):
                logger.info("Applying shape mask...")
                umap_coords = self._apply_shape_mask(umap_coords, request.shape_mask_path)
            
            # データポイントを作成
            data_points = self._create_data_points(
                processed_df, umap_coords, cluster_labels, tags, request.column_mapping
            )
            
            # クラスタ情報を生成
            clusters = self._generate_cluster_info(data_points, cluster_labels)
            
            # 結果を保存
            result = {
                "data_points": data_points,
                "clusters": clusters,
                "tags": list(set([tag for point in data_points for tag in point.tags])),
                "config": self.current_config
            }
            
            self._save_results(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _load_current_data(self) -> Optional[pd.DataFrame]:
        """現在のデータを読み込み（実際の実装では、アップロードされたデータを管理）"""
        # 実際の実装では、アップロードされたデータをセッションやデータベースから取得
        # ここでは仮の実装
        return None
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """テキストの埋め込みベクトルを生成"""
        try:
            model = self._get_sentence_model()
            embeddings = model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _apply_umap(self, embeddings: np.ndarray, umap_params: Dict[str, Any]) -> np.ndarray:
        """UMAP次元圧縮を適用"""
        try:
            # デフォルトパラメータとマージ
            params = self.config.default_umap_params.copy()
            params.update(umap_params)
            
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=params.get("n_neighbors", 15),
                min_dist=params.get("min_dist", 0.1),
                random_state=params.get("random_state", 42)
            )
            
            coords = reducer.fit_transform(embeddings)
            return coords
        except Exception as e:
            logger.error(f"UMAP failed: {e}")
            raise
    
    def _perform_clustering(
        self, 
        embeddings: np.ndarray, 
        method: str, 
        hdbscan_params: Dict[str, Any],
        kmeans_params: Dict[str, Any]
    ) -> np.ndarray:
        """クラスタリングを実行"""
        try:
            if method == "hdbscan":
                params = self.config.default_hdbscan_params.copy()
                params.update(hdbscan_params)
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=params.get("min_cluster_size", 15),
                    min_samples=params.get("min_samples", 5)
                )
                cluster_labels = clusterer.fit_predict(embeddings)
                
            elif method == "kmeans":
                params = self.config.default_kmeans_params.copy()
                params.update(kmeans_params)
                
                clusterer = KMeans(
                    n_clusters=params.get("n_clusters", 8),
                    random_state=params.get("random_state", 42)
                )
                cluster_labels = clusterer.fit_predict(embeddings)
                
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            return cluster_labels
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise
    
    def _generate_and_apply_tags(
        self, 
        texts: List[str], 
        tag_rules: List[TagRule]
    ) -> List[List[str]]:
        """タグを生成して適用"""
        try:
            # 各テキストからタグを生成
            all_tags = []
            for text in texts:
                # 基本的なキーワード抽出
                tokens = preprocess_text(text)
                # 頻度の高いトークンをタグとして使用
                from collections import Counter
                token_counts = Counter(tokens)
                text_tags = [token for token, count in token_counts.most_common(5) if count > 1]
                all_tags.append(text_tags)
            
            return all_tags
        except Exception as e:
            logger.error(f"Tag generation failed: {e}")
            return [[] for _ in texts]
    
    def _apply_shape_mask(self, coords: np.ndarray, mask_path: str) -> np.ndarray:
        """図形マスクを適用（簡易実装）"""
        # 実際の実装では、SVGパスや画像マスクを解析して座標を調整
        # ここでは簡易的に座標を正規化
        coords_normalized = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
        return coords_normalized
    
    def _create_data_points(
        self, 
        df: pd.DataFrame, 
        coords: np.ndarray, 
        cluster_labels: np.ndarray,
        tags: List[List[str]],
        column_mapping: ColumnMapping
    ) -> List[DataPoint]:
        """データポイントを作成"""
        data_points = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            point_id = row.get(column_mapping.id_column, i) if column_mapping.id_column else i
            group = row.get(column_mapping.group_column) if column_mapping.group_column else None
            
            data_point = DataPoint(
                id=point_id,
                text=row[column_mapping.text_column],
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                cluster_id=int(cluster_labels[i]),
                tags=tags[i] if i < len(tags) else [],
                group=group
            )
            data_points.append(data_point)
        
        return data_points
    
    def _generate_cluster_info(
        self, 
        data_points: List[DataPoint], 
        cluster_labels: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """クラスタ情報を生成"""
        clusters = {}
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # ノイズクラスタ
                continue
                
            cluster_points = [p for p in data_points if p.cluster_id == cluster_id]
            
            # クラスタ内のタグを集計
            all_tags = []
            for point in cluster_points:
                all_tags.extend(point.tags)
            
            from collections import Counter
            tag_counts = Counter(all_tags)
            
            clusters[cluster_id] = {
                "size": len(cluster_points),
                "top_tags": [tag for tag, count in tag_counts.most_common(5)],
                "center_x": np.mean([p.x for p in cluster_points]),
                "center_y": np.mean([p.y for p in cluster_points])
            }
        
        return clusters
    
    def _save_results(self, result: Dict[str, Any]) -> None:
        """結果を保存"""
        try:
            results_dir = Path(self.config.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 結果をJSONで保存
            result_file = results_dir / "latest_analysis.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Results saved to {result_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_current_results(self) -> Optional[Dict[str, Any]]:
        """現在の解析結果を取得"""
        try:
            result_file = Path(self.config.results_dir) / "latest_analysis.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load current results: {e}")
        return None
