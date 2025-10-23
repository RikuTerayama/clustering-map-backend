import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from app.models.schemas import DataPoint
from app.models.config import AppConfig

logger = logging.getLogger(__name__)

# 日本語フォントの設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


class SimpleExportService:
    """軽量版エクスポートサービス"""
    
    def __init__(self):
        self.config = AppConfig()
    
    def export_pdf(self, data_points: List[DataPoint], output_path: str, title: str = "Clustering Map") -> bool:
        """PDFエクスポート（軽量版）"""
        try:
            # データポイントをDataFrameに変換
            df = pd.DataFrame([dp.dict() for dp in data_points])
            
            # 図の作成
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # クラスタごとに色分けしてプロット
            colors = plt.cm.Set3(np.linspace(0, 1, len(df['cluster_id'].unique())))
            
            for i, cluster_id in enumerate(df['cluster_id'].unique()):
                cluster_data = df[df['cluster_id'] == cluster_id]
                ax.scatter(
                    cluster_data['x'], 
                    cluster_data['y'],
                    c=[colors[i]], 
                    label=f'Cluster {cluster_id + 1}',
                    alpha=0.7,
                    s=50
                )
            
            # グラフの設定
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # レイアウトの調整
            plt.tight_layout()
            
            # PDFとして保存
            plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"PDF exported successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            return False
    
    def export_png(self, data_points: List[DataPoint], output_path: str, title: str = "Clustering Map") -> bool:
        """PNGエクスポート（軽量版）"""
        try:
            # データポイントをDataFrameに変換
            df = pd.DataFrame([dp.dict() for dp in data_points])
            
            # 図の作成
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # クラスタごとに色分けしてプロット
            colors = plt.cm.Set3(np.linspace(0, 1, len(df['cluster_id'].unique())))
            
            for i, cluster_id in enumerate(df['cluster_id'].unique()):
                cluster_data = df[df['cluster_id'] == cluster_id]
                ax.scatter(
                    cluster_data['x'], 
                    cluster_data['y'],
                    c=[colors[i]], 
                    label=f'Cluster {cluster_id + 1}',
                    alpha=0.7,
                    s=50
                )
            
            # グラフの設定
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # レイアウトの調整
            plt.tight_layout()
            
            # PNGとして保存
            plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"PNG exported successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"PNG export failed: {e}")
            return False
