import pytest
import tempfile
import os
import json
from app.utils.config_utils import (
    ConfigManager, ResultManager, create_analysis_config, validate_config
)


class TestConfigUtils:
    """設定管理ユーティリティのテスト"""
    
    def test_config_manager(self):
        """設定管理のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # テスト用の設定
            test_config = {
                "column_mapping": {"text_column": "text", "id_column": "id"},
                "cluster_method": "hdbscan",
                "cluster_params": {"min_cluster_size": 5},
                "umap_params": {"n_neighbors": 15}
            }
            
            # 設定を保存
            saved_path = config_manager.save_analysis_config(test_config, "test_config")
            assert os.path.exists(saved_path)
            
            # 設定を読み込み
            loaded_config = config_manager.load_analysis_config("test_config")
            assert loaded_config == test_config
            
            # 設定一覧を取得
            configs = config_manager.list_configs()
            assert len(configs) == 1
            assert configs[0]["name"] == "test_config"
            
            # 設定を削除
            success = config_manager.delete_config("test_config")
            assert success == True
            assert not os.path.exists(saved_path)
    
    def test_result_manager(self):
        """結果管理のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result_manager = ResultManager(temp_dir)
            
            # テスト用の結果
            test_result = {
                "data_points": [
                    {"id": 1, "x": 0.1, "y": 0.2, "text": "テスト1"},
                    {"id": 2, "x": 0.3, "y": 0.4, "text": "テスト2"}
                ],
                "clusters": {0: {"size": 2, "center_x": 0.2, "center_y": 0.3}},
                "tags": ["テスト", "サンプル"]
            }
            
            # 結果を保存
            saved_path = result_manager.save_analysis_result(test_result, "test_result")
            assert os.path.exists(saved_path)
            
            # 結果を読み込み
            loaded_result = result_manager.load_analysis_result("test_result")
            assert loaded_result == test_result
            
            # 結果一覧を取得
            results = result_manager.list_results()
            assert len(results) == 1
            assert results[0]["name"] == "test_result"
            
            # 結果を削除
            success = result_manager.delete_result("test_result")
            assert success == True
            assert not os.path.exists(saved_path)
    
    def test_create_analysis_config(self):
        """解析設定作成のテスト"""
        config = create_analysis_config(
            column_mapping={"text_column": "text", "id_column": "id"},
            cluster_method="hdbscan",
            cluster_params={"min_cluster_size": 5},
            umap_params={"n_neighbors": 15},
            tag_rules=[],
            shape_mask_path=None
        )
        
        assert config["column_mapping"]["text_column"] == "text"
        assert config["cluster_method"] == "hdbscan"
        assert config["cluster_params"]["min_cluster_size"] == 5
        assert config["umap_params"]["n_neighbors"] == 15
        assert "created_at" in config
    
    def test_validate_config(self):
        """設定検証のテスト"""
        # 正常な設定
        valid_config = {
            "column_mapping": {"text_column": "text", "id_column": "id"},
            "cluster_method": "hdbscan",
            "cluster_params": {"min_cluster_size": 5},
            "umap_params": {"n_neighbors": 15}
        }
        assert validate_config(valid_config) == True
        
        # 必須フィールドが不足
        invalid_config = {
            "column_mapping": {"text_column": "text"},
            "cluster_method": "hdbscan"
            # cluster_params, umap_params が不足
        }
        assert validate_config(invalid_config) == False
        
        # text_columnが空
        invalid_config2 = {
            "column_mapping": {"text_column": ""},
            "cluster_method": "hdbscan",
            "cluster_params": {"min_cluster_size": 5},
            "umap_params": {"n_neighbors": 15}
        }
        assert validate_config(invalid_config2) == False
        
        # 無効なクラスタリング手法
        invalid_config3 = {
            "column_mapping": {"text_column": "text"},
            "cluster_method": "invalid_method",
            "cluster_params": {"min_cluster_size": 5},
            "umap_params": {"n_neighbors": 15}
        }
        assert validate_config(invalid_config3) == False
