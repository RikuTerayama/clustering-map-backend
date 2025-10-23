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


class ExportService:
    """エクスポートサービス"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.current_data = None
    
    def export_to_pdf(self, width: int = 800, height: int = 600, 
                     title: str = "クラスタリングマップ", 
                     show_legend: bool = True) -> str:
        """PDFエクスポート"""
        try:
            # 現在の解析結果を取得
            results = self._load_current_results()
            if not results:
                raise ValueError("解析結果が見つかりません。先に解析を実行してください。")
            
            # 図を作成
            fig, ax = self._create_plot(results, width, height, title, show_legend)
            
            # PDFファイルに保存
            output_path = Path(self.config.results_dir) / f"clustering_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            logger.info(f"PDF exported to {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            raise
    
    def export_to_png(self, width: int = 800, height: int = 600,
                     title: str = "クラスタリングマップ",
                     show_legend: bool = True) -> str:
        """PNGエクスポート"""
        try:
            # 現在の解析結果を取得
            results = self._load_current_results()
            if not results:
                raise ValueError("解析結果が見つかりません。先に解析を実行してください。")
            
            # 図を作成
            fig, ax = self._create_plot(results, width, height, title, show_legend)
            
            # PNGファイルに保存
            output_path = Path(self.config.results_dir) / f"clustering_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            logger.info(f"PNG exported to {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"PNG export failed: {e}")
            raise
    
    def _load_current_results(self) -> Optional[Dict[str, Any]]:
        """現在の解析結果を読み込み"""
        try:
            result_file = Path(self.config.results_dir) / "latest_analysis.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load current results: {e}")
        return None
    
    def _create_plot(self, results: Dict[str, Any], width: int, height: int,
                    title: str, show_legend: bool) -> tuple:
        """プロットを作成"""
        try:
            # データポイントを取得
            data_points = results.get("data_points", [])
            if not data_points:
                raise ValueError("データポイントが見つかりません")
            
            # 図のサイズを設定
            fig_width = width / 100  # matplotlibはインチ単位
            fig_height = height / 100
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # データポイントをプロット
            self._plot_data_points(ax, data_points)
            
            # タイトルを設定
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # 軸ラベルを設定
            ax.set_xlabel("UMAP 1", fontsize=12)
            ax.set_ylabel("UMAP 2", fontsize=12)
            
            # グリッドを表示
            ax.grid(True, alpha=0.3)
            
            # 凡例を表示
            if show_legend:
                self._add_legend(ax, data_points)
            
            # レイアウトを調整
            plt.tight_layout()
            
            return fig, ax
        
        except Exception as e:
            logger.error(f"Plot creation failed: {e}")
            raise
    
    def _plot_data_points(self, ax, data_points: List[Dict[str, Any]]):
        """データポイントをプロット"""
        try:
            # クラスタごとに色分け
            cluster_colors = {}
            color_palette = plt.cm.Set3(np.linspace(0, 1, 12))
            
            for i, point in enumerate(data_points):
                cluster_id = point.get("cluster_id", -1)
                
                if cluster_id not in cluster_colors:
                    cluster_colors[cluster_id] = color_palette[len(cluster_colors) % len(color_palette)]
                
                color = cluster_colors[cluster_id]
                
                # ノイズクラスタ（-1）は灰色
                if cluster_id == -1:
                    color = 'gray'
                    alpha = 0.5
                else:
                    alpha = 0.7
                
                # 点をプロット
                ax.scatter(
                    point["x"], point["y"],
                    c=[color], alpha=alpha, s=30,
                    edgecolors='white', linewidth=0.5
                )
            
            # クラスタの中心をプロット
            self._plot_cluster_centers(ax, data_points, cluster_colors)
        
        except Exception as e:
            logger.error(f"Data points plotting failed: {e}")
            raise
    
    def _plot_cluster_centers(self, ax, data_points: List[Dict[str, Any]], cluster_colors: Dict[int, str]):
        """クラスタの中心をプロット"""
        try:
            # クラスタごとにデータをグループ化
            clusters = {}
            for point in data_points:
                cluster_id = point.get("cluster_id", -1)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(point)
            
            # 各クラスタの中心を計算してプロット
            for cluster_id, points in clusters.items():
                if cluster_id == -1:  # ノイズクラスタはスキップ
                    continue
                
                if len(points) < 2:  # 点が少なすぎる場合はスキップ
                    continue
                
                center_x = np.mean([p["x"] for p in points])
                center_y = np.mean([p["y"] for p in points])
                
                # 中心点をプロット
                ax.scatter(
                    center_x, center_y,
                    c=[cluster_colors[cluster_id]], s=100,
                    marker='x', linewidth=3,
                    edgecolors='black'
                )
                
                # クラスタIDを表示
                ax.annotate(
                    f'C{cluster_id}',
                    (center_x, center_y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )
        
        except Exception as e:
            logger.error(f"Cluster centers plotting failed: {e}")
    
    def _add_legend(self, ax, data_points: List[Dict[str, Any]]):
        """凡例を追加"""
        try:
            # クラスタごとの情報を取得
            clusters = {}
            for point in data_points:
                cluster_id = point.get("cluster_id", -1)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(point)
            
            # 凡例用のラベルと色を作成
            legend_elements = []
            color_palette = plt.cm.Set3(np.linspace(0, 1, 12))
            
            for i, (cluster_id, points) in enumerate(clusters.items()):
                if cluster_id == -1:  # ノイズクラスタ
                    label = f"ノイズ ({len(points)}点)"
                    color = 'gray'
                else:
                    label = f"クラスタ {cluster_id} ({len(points)}点)"
                    color = color_palette[i % len(color_palette)]
                
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=8, label=label)
                )
            
            # 凡例を追加
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.0, 1.0), fontsize=10)
        
        except Exception as e:
            logger.error(f"Legend addition failed: {e}")
    
    def export_data_to_csv(self, output_path: Optional[str] = None) -> str:
        """データをCSVでエクスポート"""
        try:
            results = self._load_current_results()
            if not results:
                raise ValueError("解析結果が見つかりません")
            
            data_points = results.get("data_points", [])
            if not data_points:
                raise ValueError("データポイントが見つかりません")
            
            # DataFrameに変換
            df = pd.DataFrame(data_points)
            
            # 出力パスを設定
            if output_path is None:
                output_path = Path(self.config.results_dir) / f"clustering_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # CSVで保存
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"Data exported to CSV: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise
