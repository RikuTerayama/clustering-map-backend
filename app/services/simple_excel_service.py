import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import json
import os
from pathlib import Path
import re
from collections import Counter

from app.models.schemas import TagCandidate, TagRule, ColumnMapping
from app.models.config import AppConfig

logger = logging.getLogger(__name__)


class SimpleExcelService:
    """軽量版Excelファイル処理サービス（重いライブラリなし）"""
    
    def __init__(self):
        self.config = AppConfig()
    
    def process_excel_file(self, file_path: str, column_mapping: ColumnMapping) -> Dict[str, Any]:
        """Excelファイルの処理（軽量版）"""
        try:
            # Excelファイルを読み込み
            df = pd.read_excel(file_path)
            
            # 列マッピングに基づいてデータを抽出
            texts = df[column_mapping.text_column].fillna('').astype(str).tolist()
            groups = df[column_mapping.group_column].fillna('').astype(str).tolist() if column_mapping.group_column else None
            ids = df[column_mapping.id_column].fillna('').astype(str).tolist() if column_mapping.id_column else None
            
            # 基本的なタグ候補を生成（ルールベース）
            tag_candidates = self._generate_simple_tags(texts)
            
            return {
                "texts": texts,
                "groups": groups,
                "ids": ids,
                "tag_candidates": tag_candidates,
                "total_responses": len(texts),
                "columns": list(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Excel processing failed: {e}")
            raise Exception(f"Excelファイルの処理中にエラーが発生しました: {str(e)}")
    
    def _generate_simple_tags(self, texts: List[str]) -> List[TagCandidate]:
        """基本的なタグ候補を生成（ルールベース）"""
        # 基本的なキーワード抽出
        all_words = []
        for text in texts:
            # 簡単な前処理
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # 頻出単語をカウント
        word_counts = Counter(all_words)
        
        # タグ候補を生成
        tag_candidates = []
        for word, count in word_counts.most_common(20):  # 上位20個
            if len(word) > 2 and count > 1:  # 2文字以上、2回以上出現
                tag_candidates.append(TagCandidate(
                    text=word,
                    score=count / len(texts),
                    category="自動生成"
                ))
        
        return tag_candidates
    
    def update_tag_rules(self, rules: List[TagRule]) -> bool:
        """タグルールの更新"""
        try:
            rules_data = [rule.dict() for rule in rules]
            with open(self.config.tag_rules_path, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Tag rules update failed: {e}")
            return False
