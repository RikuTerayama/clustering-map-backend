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
            # 基本的なテキスト分析
            texts = request.texts
            n_clusters = min(5, len(texts) // 3) if len(texts) > 3 else 1
            
            # TF-IDFベクトル化
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # クラスタリング（KMeans）
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
            else:
                cluster_labels = [0] * len(texts)
            
            # データポイントを生成
            data_points = []
            for i, text in enumerate(texts):
                # 基本的なテキスト分析
                word_count = len(text.split()) if text else 0
                char_count = len(text) if text else 0
                
                # 簡単な座標生成（クラスタに基づく）
                cluster_id = cluster_labels[i]
                x = np.random.uniform(0, 1) + cluster_id * 0.2
                y = np.random.uniform(0, 1) + cluster_id * 0.2
                
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
                        "department": request.group_column and request.groups[i] if request.groups else None
                    }
                )
                data_points.append(data_point)
            
            # クラスタ情報を生成
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_points = [dp for dp in data_points if dp.cluster_id == cluster_id]
                clusters.append({
                    "id": cluster_id,
                    "name": f"Cluster {cluster_id + 1}",
                    "count": len(cluster_points)
                })
            
            return {
                "data_points": [dp.model_dump() for dp in data_points],
                "clusters": clusters,
                "tags": self._get_all_tags(data_points),
                "statistics": {
                    "total_responses": len(data_points),
                    "average_word_count": np.mean([dp.metadata["word_count"] for dp in data_points]),
                    "average_char_count": np.mean([dp.metadata["char_count"] for dp in data_points]),
                    "num_clusters": n_clusters
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise Exception(f"分析中にエラーが発生しました: {str(e)}")
    
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
    
    def _get_all_tags(self, data_points: List[DataPoint]) -> List[str]:
        """すべてのタグを取得"""
        all_tags = []
        for dp in data_points:
            all_tags.extend(dp.tags)
        return list(set(all_tags))
