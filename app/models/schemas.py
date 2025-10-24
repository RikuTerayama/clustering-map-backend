from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class ClusterMethod(str, Enum):
    HDBSCAN = "hdbscan"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"


class ColumnMapping(BaseModel):
    text_column: str = Field(..., description="本文列の名前")
    id_column: Optional[str] = Field(None, description="ID列の名前")
    group_column: Optional[str] = Field(None, description="グループ列の名前")


class TagRule(BaseModel):
    """タグルールの定義"""
    key: str = Field(..., description="正規化後のタグ名")
    synonyms: List[str] = Field(..., description="同義語のリスト")
    category: Optional[str] = Field(None, description="タグカテゴリ")


class TagCandidate(BaseModel):
    """タグ候補"""
    text: str = Field(..., description="タグテキスト")
    score: float = Field(..., description="重要度スコア")
    category: Optional[str] = Field(None, description="タグカテゴリ")
    count: int = Field(1, description="出現回数")


class UploadResponse(BaseModel):
    """アップロード応答"""
    success: bool
    message: str
    columns: List[str] = Field(..., description="利用可能な列名")
    sample_data: List[Dict[str, Any]] = Field(..., description="サンプルデータ（最初の5行）")
    tag_candidates: List[TagCandidate] = Field(..., description="タグ候補")


class AnalysisRequest(BaseModel):
    """解析リクエスト"""
    column_mapping: ColumnMapping
    tag_rules: List[TagRule] = Field(default_factory=list)
    cluster_method: ClusterMethod = ClusterMethod.HDBSCAN
    hdbscan_params: Dict[str, Any] = Field(default_factory=lambda: {
        "min_cluster_size": 15,
        "min_samples": 5
    })
    kmeans_params: Dict[str, Any] = Field(default_factory=lambda: {
        "n_clusters": 8
    })
    umap_params: Dict[str, Any] = Field(default_factory=lambda: {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "random_state": 42
    })
    shape_mask_path: Optional[str] = Field(None, description="図形マスクのパス")
    config: Optional[Dict[str, Any]] = Field(None, description="解析設定")


class DataPoint(BaseModel):
    """データポイント"""
    id: Union[str, int]
    text: str
    x: float
    y: float
    cluster_id: int
    tags: List[str]
    group: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """解析応答"""
    success: bool
    message: str
    data_points: List[DataPoint]
    clusters: Dict[int, Dict[str, Any]] = Field(..., description="クラスタ情報")
    tags: List[str] = Field(..., description="使用されたタグ一覧")
    config: Dict[str, Any] = Field(..., description="使用された設定")


class ExportRequest(BaseModel):
    """エクスポートリクエスト"""
    format: str = Field(..., description="エクスポート形式 (pdf/png)")
    width: int = Field(800, description="画像幅")
    height: int = Field(600, description="画像高さ")
    title: str = Field("クラスタリングマップ", description="タイトル")
    show_legend: bool = Field(True, description="凡例を表示するか")
    show_labels: bool = Field(False, description="ラベルを表示するか")


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    success: bool = False
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
