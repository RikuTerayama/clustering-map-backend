import pytest
import pandas as pd
import tempfile
import os
from app.utils.file_utils import (
    read_excel_file, validate_excel_columns, get_sample_data,
    save_results, load_results, is_valid_excel_file
)


class TestFileUtils:
    """ファイル処理ユーティリティのテスト"""
    
    def test_read_excel_file(self):
        """Excelファイル読み込みのテスト"""
        # テスト用のDataFrameを作成
        test_data = {
            'id': [1, 2, 3],
            'text': ['テスト1', 'テスト2', 'テスト3'],
            'category': ['A', 'B', 'A']
        }
        df = pd.DataFrame(test_data)
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df.to_excel(tmp.name, index=False, engine='openpyxl')
            tmp_path = tmp.name
        
        try:
            # ファイルを読み込み
            result_df = read_excel_file(tmp_path)
            
            # 結果を検証
            assert len(result_df) == 3
            assert list(result_df.columns) == ['id', 'text', 'category']
            assert result_df.iloc[0]['text'] == 'テスト1'
        finally:
            # 一時ファイルを削除
            os.unlink(tmp_path)
    
    def test_validate_excel_columns(self):
        """Excel列検証のテスト"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['テスト1', 'テスト2', 'テスト3'],
            'category': ['A', 'B', 'A']
        })
        
        # 正常なケース
        assert validate_excel_columns(df, ['id', 'text']) == True
        
        # 存在しない列があるケース
        with pytest.raises(ValueError):
            validate_excel_columns(df, ['id', 'nonexistent'])
    
    def test_get_sample_data(self):
        """サンプルデータ取得のテスト"""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'text': ['テスト1', 'テスト2', 'テスト3', 'テスト4', 'テスト5', 'テスト6']
        })
        
        # デフォルト（5行）
        sample = get_sample_data(df)
        assert len(sample) == 5
        
        # 指定した行数
        sample = get_sample_data(df, 3)
        assert len(sample) == 3
        
        # データフレームより多い行数を指定
        sample = get_sample_data(df, 10)
        assert len(sample) == 6  # 全行が返される
    
    def test_save_and_load_results(self):
        """結果保存・読み込みのテスト"""
        test_data = {
            'data_points': [
                {'id': 1, 'x': 0.1, 'y': 0.2, 'text': 'テスト1'},
                {'id': 2, 'x': 0.3, 'y': 0.4, 'text': 'テスト2'}
            ],
            'clusters': {0: {'size': 2, 'center_x': 0.2, 'center_y': 0.3}},
            'tags': ['テスト', 'サンプル']
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # 結果を保存
            save_results(test_data, tmp_path)
            
            # ファイルが作成されたことを確認
            assert os.path.exists(tmp_path)
            
            # 結果を読み込み
            loaded_data = load_results(tmp_path)
            
            # データが正しく保存・読み込みされたことを確認
            assert loaded_data['data_points'] == test_data['data_points']
            assert loaded_data['clusters'] == test_data['clusters']
            assert loaded_data['tags'] == test_data['tags']
        finally:
            # 一時ファイルを削除
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_is_valid_excel_file(self):
        """Excelファイル形式検証のテスト"""
        # 有効な拡張子
        assert is_valid_excel_file('test.xlsx') == True
        assert is_valid_excel_file('test.xls') == True
        
        # 無効な拡張子
        assert is_valid_excel_file('test.txt') == False
        assert is_valid_excel_file('test.csv') == False
        assert is_valid_excel_file('test') == False
        
        # 大文字小文字の区別
        assert is_valid_excel_file('test.XLSX') == True
        assert is_valid_excel_file('test.XLS') == True
