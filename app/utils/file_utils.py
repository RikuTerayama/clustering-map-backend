import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def read_excel_file(file_path: str) -> pd.DataFrame:
    """Excelファイルを読み込む"""
    try:
        # 複数のシートがある場合は最初のシートを読み込み
        df = pd.read_excel(file_path, engine='openpyxl')
        logger.info(f"Excel file loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        raise ValueError(f"Excelファイルの読み込みに失敗しました: {e}")


def validate_excel_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Excelファイルの列を検証"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"必要な列が見つかりません: {missing_columns}")
    return True


def get_sample_data(df: pd.DataFrame, n_rows: int = 5) -> List[Dict[str, Any]]:
    """サンプルデータを取得"""
    sample_df = df.head(n_rows)
    return sample_df.to_dict('records')


def save_results(data: Dict[str, Any], file_path: str) -> None:
    """結果をJSONファイルに保存"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_results(file_path: str) -> Dict[str, Any]:
    """結果をJSONファイルから読み込み"""
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_file_extension(file_path: str) -> str:
    """ファイル拡張子を取得"""
    return Path(file_path).suffix.lower()


def is_valid_excel_file(file_path: str) -> bool:
    """有効なExcelファイルかチェック"""
    valid_extensions = ['.xlsx', '.xls']
    return get_file_extension(file_path) in valid_extensions


def create_temp_file(content: bytes, suffix: str = '.xlsx') -> str:
    """一時ファイルを作成"""
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


def cleanup_temp_file(file_path: str) -> None:
    """一時ファイルを削除"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
