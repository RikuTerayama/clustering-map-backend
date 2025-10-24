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
    
    def generate_tag_candidates(self, df: pd.DataFrame) -> List[TagCandidate]:
        """DataFrameからタグ候補を生成"""
        try:
            # テキスト列を自動検出（最初の列または'自由記述'列）
            text_column = None
            for col in df.columns:
                if '自由記述' in str(col) or 'text' in str(col).lower() or 'comment' in str(col).lower():
                    text_column = col
                    break
            
            if text_column is None:
                # 最初の列を使用
                text_column = df.columns[0]
            
            logger.info(f"Using text column: {text_column}")
            
            # テキストデータを取得
            texts = df[text_column].fillna('').astype(str).tolist()
            
            return self._generate_simple_tags(texts)
            
        except Exception as e:
            logger.error(f"Tag candidate generation failed: {e}")
            # フォールバック: 空のリストを返す
            return []

    def _generate_simple_tags(self, texts: List[str]) -> List[TagCandidate]:
        """ビジネス文脈に沿ったタグ候補を生成"""
        try:
            # ビジネス関連のキーワード辞書
            business_keywords = {
                "満足度": ["満足", "良い", "素晴らしい", "優秀", "完璧", "最高", "気に入り", "おすすめ"],
                "不満": ["不満", "悪い", "問題", "困る", "改善", "要望", "残念", "残念"],
                "価格": ["価格", "料金", "コスト", "費用", "安い", "高い", "値段", "価値"],
                "サービス": ["サービス", "対応", "サポート", "支援", "ヘルプ", "案内", "説明"],
                "機能": ["機能", "性能", "仕様", "特徴", "使いやすさ", "操作性", "インターフェース"],
                "品質": ["品質", "質", "精度", "正確性", "信頼性", "安定性", "耐久性"],
                "デザイン": ["デザイン", "見た目", "外観", "美しい", "スタイル", "レイアウト", "UI"],
                "速度": ["速度", "速い", "遅い", "レスポンス", "処理時間", "効率", "パフォーマンス"],
                "使いやすさ": ["使いやすい", "簡単", "直感的", "分かりやすい", "操作", "手順"],
                "カスタマーサポート": ["サポート", "対応", "問い合わせ", "相談", "ヘルプ", "フォロー"]
            }
            
            # テキストからキーワードを抽出
            all_words = []
            for text in texts:
                if not text or text.strip() == '':
                    continue
                # 簡単な前処理
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend(words)
            
            if not all_words:
                return []
            
            # 頻出単語をカウント
            word_counts = Counter(all_words)
            
            # ビジネスカテゴリベースのタグ候補を生成
            tag_candidates = []
            
            # 各ビジネスカテゴリについて、関連キーワードの出現頻度を計算
            for category, keywords in business_keywords.items():
                category_score = 0
                category_count = 0
                
                for keyword in keywords:
                    if keyword in word_counts:
                        category_score += word_counts[keyword]
                        category_count += word_counts[keyword]
                
                if category_count > 0:
                    # カテゴリ全体のスコア
                    tag_candidates.append(TagCandidate(
                        text=category,
                        score=category_score / len(texts),
                        category="ビジネスカテゴリ",
                        count=category_count
                    ))
            
            # 個別の頻出キーワードも追加（ビジネス関連のもののみ）
            business_related_words = set()
            for keywords in business_keywords.values():
                business_related_words.update(keywords)
            
            for word, count in word_counts.most_common(30):
                if (len(word) > 2 and count > 1 and 
                    (word in business_related_words or 
                     any(business_word in word for business_word in business_related_words))):
                    tag_candidates.append(TagCandidate(
                        text=word,
                        score=count / len(texts),
                        category="キーワード",
                        count=count
                    ))
            
            # スコア順でソート
            tag_candidates.sort(key=lambda x: x.score, reverse=True)
            
            # 上位20個を返す
            result = tag_candidates[:20]
            logger.info(f"Generated {len(result)} business-context tag candidates")
            return result
            
        except Exception as e:
            logger.error(f"Business tag generation failed: {e}")
            return []
    
    def get_tag_rules(self) -> List[Dict[str, Any]]:
        """タグルールを取得"""
        try:
            rules_path = os.path.join(self.config.data_dir, "tags", "tag_rules.json")
            if os.path.exists(rules_path):
                with open(rules_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Tag rules loading failed: {e}")
            return []

    def update_tag_rules(self, rules: List[TagRule]) -> bool:
        """タグルールの更新"""
        try:
            rules_data = [rule.model_dump() for rule in rules]
            rules_path = os.path.join(self.config.data_dir, "tags", "tag_rules.json")
            os.makedirs(os.path.dirname(rules_path), exist_ok=True)
            with open(rules_path, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Tag rules update failed: {e}")
            return False
