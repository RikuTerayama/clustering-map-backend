import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, config_dir: str = "data/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_analysis_config(self, config: Dict[str, Any], name: Optional[str] = None) -> str:
        """解析設定を保存"""
        try:
            if name is None:
                name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            config_file = self.config_dir / f"{name}.json"
            
            # メタデータを追加
            config_with_meta = {
                "name": name,
                "created_at": datetime.now().isoformat(),
                "version": "0.1.0",
                "config": config
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_with_meta, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Config saved: {config_file}")
            return str(config_file)
        
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def load_analysis_config(self, name: str) -> Optional[Dict[str, Any]]:
        """解析設定を読み込み"""
        try:
            config_file = self.config_dir / f"{name}.json"
            
            if not config_file.exists():
                return None
            
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get("config")
        
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return None
    
    def list_configs(self) -> list:
        """保存された設定一覧を取得"""
        try:
            configs = []
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    configs.append({
                        "name": data.get("name", config_file.stem),
                        "created_at": data.get("created_at", ""),
                        "version": data.get("version", ""),
                        "file_path": str(config_file)
                    })
                except Exception as e:
                    logger.warning(f"Failed to read config file {config_file}: {e}")
                    continue
            
            # 作成日時でソート
            configs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return configs
        
        except Exception as e:
            logger.error(f"Failed to list configs: {e}")
            return []
    
    def delete_config(self, name: str) -> bool:
        """設定を削除"""
        try:
            config_file = self.config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
                logger.info(f"Config deleted: {config_file}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"Failed to delete config: {e}")
            return False


class ResultManager:
    """結果管理クラス"""
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_analysis_result(self, result: Dict[str, Any], name: Optional[str] = None) -> str:
        """解析結果を保存"""
        try:
            if name is None:
                name = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result_file = self.results_dir / f"{name}.json"
            
            # メタデータを追加
            result_with_meta = {
                "name": name,
                "created_at": datetime.now().isoformat(),
                "version": "0.1.0",
                "result": result
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_with_meta, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Result saved: {result_file}")
            return str(result_file)
        
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            raise
    
    def load_analysis_result(self, name: str) -> Optional[Dict[str, Any]]:
        """解析結果を読み込み"""
        try:
            result_file = self.results_dir / f"{name}.json"
            
            if not result_file.exists():
                return None
            
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get("result")
        
        except Exception as e:
            logger.error(f"Failed to load result: {e}")
            return None
    
    def list_results(self) -> list:
        """保存された結果一覧を取得"""
        try:
            results = []
            for result_file in self.results_dir.glob("*.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    results.append({
                        "name": data.get("name", result_file.stem),
                        "created_at": data.get("created_at", ""),
                        "version": data.get("version", ""),
                        "file_path": str(result_file)
                    })
                except Exception as e:
                    logger.warning(f"Failed to read result file {result_file}: {e}")
                    continue
            
            # 作成日時でソート
            results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return results
        
        except Exception as e:
            logger.error(f"Failed to list results: {e}")
            return []
    
    def delete_result(self, name: str) -> bool:
        """結果を削除"""
        try:
            result_file = self.results_dir / f"{name}.json"
            if result_file.exists():
                result_file.unlink()
                logger.info(f"Result deleted: {result_file}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"Failed to delete result: {e}")
            return False


def create_analysis_config(
    column_mapping: Dict[str, str],
    cluster_method: str,
    cluster_params: Dict[str, Any],
    umap_params: Dict[str, Any],
    tag_rules: list,
    shape_mask_path: Optional[str] = None
) -> Dict[str, Any]:
    """解析設定を作成"""
    return {
        "column_mapping": column_mapping,
        "cluster_method": cluster_method,
        "cluster_params": cluster_params,
        "umap_params": umap_params,
        "tag_rules": tag_rules,
        "shape_mask_path": shape_mask_path,
        "created_at": datetime.now().isoformat()
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """設定の妥当性を検証"""
    required_fields = ["column_mapping", "cluster_method", "cluster_params", "umap_params"]
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field: {field}")
            return False
    
    # 列マッピングの検証
    column_mapping = config["column_mapping"]
    if "text_column" not in column_mapping or not column_mapping["text_column"]:
        logger.error("text_column is required")
        return False
    
    # クラスタリング手法の検証
    valid_methods = ["hdbscan", "kmeans", "dbscan"]
    if config["cluster_method"] not in valid_methods:
        logger.error(f"Invalid cluster method: {config['cluster_method']}")
        return False
    
    return True
