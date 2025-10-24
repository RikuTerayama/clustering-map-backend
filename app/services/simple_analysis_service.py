import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from pathlib import Path
import re
from collections import Counter

from app.models.schemas import (
    AnalysisRequest, DataPoint, TagRule, ColumnMapping
)
from app.models.config import AppConfig

logger = logging.getLogger(__name__)


class SimpleAnalysisService:
    """簡素化された分析サービス（重いライブラリなし）"""
    
    def __init__(self):
        self.config = AppConfig()
    
    def analyze_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """データの分析（簡素化版）"""
        try:
            logger.info("Starting analysis...")
            logger.info(f"Request attributes: {dir(request)}")
            logger.info(f"Request type: {type(request)}")
            logger.info(f"Request config: {getattr(request, 'config', 'NOT_FOUND')}")
            logger.info(f"Self config: {self.config}")
            
            # ダミーデータを生成（実際の実装では、アップロードされたデータを使用）
            texts = [
                "このサービスはとても使いやすく、機能も充実しています。",
                "料金が少し高いと感じます。もう少し安くなれば利用したいです。",
                "サポートが丁寧で、問題がすぐに解決されました。",
                "機能は良いのですが、もう少しシンプルな操作ができると良いです。",
                "全体的に満足しています。継続して利用したいと思います。",
                "レスポンスが早くて助かります。使い勝手も良いです。",
                "エラーが発生することがあり、改善が必要だと思います。",
                "デザインが美しく、操作も直感的で使いやすいです。",
                "料金体系が複雑で分かりにくいです。シンプルにしてほしい。",
                "カスタマーサポートの対応が素晴らしいです。"
            ]
            
            n_clusters = min(5, len(texts) // 3) if len(texts) > 3 else 1
            logger.info(f"Number of clusters: {n_clusters}")
            
            # TF-IDFベクトル化
            logger.info("Starting TF-IDF vectorization...")
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            logger.info("TF-IDF vectorization completed")
            
            # クラスタリング（KMeans）
            logger.info("Starting clustering...")
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
            else:
                cluster_labels = [0] * len(texts)
            logger.info("Clustering completed")
            
            # 図形に基づく座標生成
            logger.info("Generating shape coordinates...")
            shape_mask = request.shape_mask_path if hasattr(request, 'shape_mask_path') else 'circle'
            coordinates = self._generate_shape_coordinates(len(texts), shape_mask)
            logger.info("Shape coordinates generated")
            
            # データポイントを生成
            logger.info("Generating data points...")
            data_points = []
            for i, text in enumerate(texts):
                # 基本的なテキスト分析
                word_count = len(text.split()) if text else 0
                char_count = len(text) if text else 0
                
                # 図形に基づく座標
                cluster_id = cluster_labels[i]
                x, y = coordinates[i]
                
                data_point = DataPoint(
                    id=str(i),
                    text=text,
                    x=float(x),
                    y=float(y),
                    cluster_id=int(cluster_id),
                    tags=self._extract_simple_tags(text),
                    metadata={
                        "word_count": word_count,
                        "char_count": char_count,
                        "department": None  # グループ情報は現在使用しない
                    }
                )
                data_points.append(data_point)
            logger.info(f"Generated {len(data_points)} data points")
            
            # クラスタ情報を生成
            logger.info("Generating cluster information...")
            clusters = {}
            for cluster_id in range(n_clusters):
                cluster_points = [dp for dp in data_points if dp.cluster_id == cluster_id]
                if cluster_points:  # 空でないクラスタのみ追加
                    clusters[cluster_id] = {
                        "size": len(cluster_points),
                        "top_tags": self._get_cluster_top_tags(cluster_points),
                        "center_x": np.mean([dp.x for dp in cluster_points]),
                        "center_y": np.mean([dp.y for dp in cluster_points])
                    }
            logger.info(f"Generated {len(clusters)} clusters")
            
            logger.info("Preparing final result...")
            return {
                "data_points": [dp.model_dump() for dp in data_points],
                "clusters": clusters,
                "tags": self._get_all_tags(data_points),
                "statistics": {
                    "total_responses": len(data_points),
                    "average_word_count": np.mean([dp.metadata["word_count"] for dp in data_points]),
                    "average_char_count": np.mean([dp.metadata["char_count"] for dp in data_points]),
                    "num_clusters": n_clusters
                },
                "config": {
                    "cluster_method": request.cluster_method,
                    "shape_mask": shape_mask,
                    "n_clusters": n_clusters,
                    "hdbscan_params": request.hdbscan_params,
                    "kmeans_params": request.kmeans_params,
                    "umap_params": request.umap_params
                }
            }
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Full traceback: {error_details}")
            raise Exception(f"分析中にエラーが発生しました: {str(e)} (詳細: {error_details})")

    def _generate_shape_coordinates(self, num_points: int, shape: str) -> List[Tuple[float, float]]:
        """指定された図形に基づいて座標を生成"""
        coordinates = []
        
        if shape == 'circle':
            # 円形配置
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                radius = 0.3 + np.random.uniform(-0.1, 0.1)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                coordinates.append((x, y))
                
        elif shape == 'square':
            # 四角形配置
            side_length = int(np.ceil(np.sqrt(num_points)))
            for i in range(num_points):
                row = i // side_length
                col = i % side_length
                x = 0.2 + (col / (side_length - 1)) * 0.6 if side_length > 1 else 0.5
                y = 0.2 + (row / (side_length - 1)) * 0.6 if side_length > 1 else 0.5
                coordinates.append((x, y))
                
        elif shape == 'triangle':
            # 三角形配置
            for i in range(num_points):
                # 三角形の頂点を基準に配置
                if i % 3 == 0:
                    x, y = 0.5, 0.8  # 上
                elif i % 3 == 1:
                    x, y = 0.2, 0.2  # 左下
                else:
                    x, y = 0.8, 0.2  # 右下
                # 少しランダムにずらす
                x += np.random.uniform(-0.1, 0.1)
                y += np.random.uniform(-0.1, 0.1)
                coordinates.append((x, y))
                
        elif shape == 'heart':
            # ハート形配置
            for i in range(num_points):
                t = 2 * np.pi * i / num_points
                x = 16 * np.sin(t)**3
                y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
                # 正規化して0-1の範囲に
                x = (x + 16) / 32
                y = (y + 20) / 40
                coordinates.append((x, y))
                
        elif shape == 'star':
            # 星形配置
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                # 5角星の形状
                radius = 0.3 + 0.1 * np.sin(5 * angle)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                coordinates.append((x, y))
                
        elif shape == 'hexagon':
            # 六角形配置
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                radius = 0.3
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                coordinates.append((x, y))
                
        else:
            # デフォルト: ランダム配置
            for i in range(num_points):
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                coordinates.append((x, y))
        
        return coordinates

    def _get_all_tags(self, data_points: List[DataPoint]) -> List[str]:
        """すべてのデータポイントからタグを抽出"""
        all_tags = set()
        for dp in data_points:
            all_tags.update(dp.tags)
        return list(all_tags)

    def _get_cluster_top_tags(self, cluster_points: List[DataPoint]) -> List[str]:
        """クラスタの上位タグを取得"""
        tag_counts = {}
        for dp in cluster_points:
            for tag in dp.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # 出現回数順でソートして上位5個を返す
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, count in sorted_tags[:5]]
    
    def _extract_simple_tags(self, text: str) -> List[str]:
        """簡単なタグ抽出"""
        if not text:
            return []
        
        # 基本的なキーワード抽出
        words = re.findall(r'\b\w+\b', text.lower())
        # 2文字以上、頻出する単語をタグとして使用
        word_counts = Counter(words)
        tags = [word for word, count in word_counts.items() if len(word) > 2 and count > 1]
        return tags[:5]  # 最大5個のタグ
