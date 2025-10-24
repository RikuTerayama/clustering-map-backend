import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path

from app.models.schemas import TagCandidate, TagRule, ColumnMapping
from app.models.config import AppConfig
from app.utils.text_utils import preprocess_text, merge_similar_tags

logger = logging.getLogger(__name__)


class ExcelService:
    """Excelファイル処理サービス"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.keybert_model = None
        self.sentence_model = None
        self.tag_rules = self._load_tag_rules()
    
    def _load_tag_rules(self) -> List[TagRule]:
        """タグルールを読み込み"""
        rules_path = os.path.join(self.config.data_dir, "tags", "tag_rules.json")
        if os.path.exists(rules_path):
            try:
                with open(rules_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [TagRule(**rule) for rule in data]
            except Exception as e:
                logger.warning(f"Failed to load tag rules: {e}")
        return []
    
    def _save_tag_rules(self) -> None:
        """タグルールを保存"""
        rules_path = os.path.join(self.config.data_dir, "tags", "tag_rules.json")
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)
        
        with open(rules_path, 'w', encoding='utf-8') as f:
            json.dump([rule.model_dump() for rule in self.tag_rules], f, ensure_ascii=False, indent=2)
    
    def _get_keybert_model(self):
        """KeyBERTモデルを取得（遅延読み込み）"""
        if self.keybert_model is None:
            try:
                self.keybert_model = KeyBERT()
            except Exception as e:
                logger.error(f"Failed to load KeyBERT model: {e}")
                raise
        return self.keybert_model
    
    def _get_sentence_model(self):
        """SentenceTransformerモデルを取得（遅延読み込み）"""
        if self.sentence_model is None:
            try:
                self.sentence_model = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                logger.error(f"Failed to load sentence model: {e}")
                raise
        return self.sentence_model
    
    def generate_tag_candidates(self, df: pd.DataFrame, text_column: str = None) -> List[TagCandidate]:
        """タグ候補を生成"""
        try:
            # テキスト列を特定
            if text_column is None:
                # 最初のテキスト列を自動選択
                text_columns = df.select_dtypes(include=['object']).columns
                if len(text_columns) == 0:
                    raise ValueError("テキスト列が見つかりません")
                text_column = text_columns[0]
            
            # テキストデータを取得
            texts = df[text_column].dropna().astype(str).tolist()
            if not texts:
                return []
            
            # KeyBERTでキーワード抽出
            keybert_model = self._get_keybert_model()
            
            all_keywords = []
            for text in texts:
                try:
                    # 各テキストからキーワードを抽出
                    keywords = keybert_model.extract_keywords(
                        text, 
                        keyphrase_ngram_range=(1, 3),
                        stop_words=None,
                        use_mmr=True,
                        diversity=0.5,
                        top_k=5
                    )
                    all_keywords.extend([kw[0] for kw in keywords])
                except Exception as e:
                    logger.warning(f"KeyBERT extraction failed for text: {e}")
                    continue
            
            # キーワードの頻度をカウント
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            
            # タグ候補を作成
            candidates = []
            for keyword, count in keyword_counts.most_common(50):  # 上位50個
                if len(keyword.strip()) > 1:  # 1文字以下は除外
                    candidates.append(TagCandidate(
                        text=keyword.strip(),
                        score=count / len(texts),  # 出現率
                        count=count
                    ))
            
            # 類似タグをマージ
            merged_keywords = merge_similar_tags([c.text for c in candidates])
            
            # マージされたタグで候補を再構築
            final_candidates = []
            for keyword in merged_keywords:
                # 元の候補から該当するものを探す
                original_candidate = next((c for c in candidates if c.text == keyword), None)
                if original_candidate:
                    final_candidates.append(original_candidate)
            
            return final_candidates[:30]  # 上位30個に制限
        
        except Exception as e:
            logger.error(f"Tag generation failed: {e}")
            return []
    
    def apply_tag_rules(self, candidates: List[TagCandidate]) -> List[TagCandidate]:
        """タグルールを適用してタグを正規化"""
        if not self.tag_rules:
            return candidates
        
        # ルール辞書を作成
        rule_dict = {}
        for rule in self.tag_rules:
            for synonym in rule.synonyms:
                rule_dict[synonym.lower()] = rule.key
        
        # タグを正規化
        normalized_candidates = []
        for candidate in candidates:
            normalized_text = rule_dict.get(candidate.text.lower(), candidate.text)
            
            # 既存の候補と統合
            existing = next((c for c in normalized_candidates if c.text == normalized_text), None)
            if existing:
                existing.count += candidate.count
                existing.score = max(existing.score, candidate.score)
            else:
                normalized_candidates.append(TagCandidate(
                    text=normalized_text,
                    score=candidate.score,
                    count=candidate.count
                ))
        
        return normalized_candidates
    
    def get_tag_rules(self) -> List[Dict[str, Any]]:
        """タグルールを取得"""
        return [rule.model_dump() for rule in self.tag_rules]
    
    def update_tag_rules(self, rules_data: Dict[str, Any]) -> None:
        """タグルールを更新"""
        try:
            self.tag_rules = [TagRule(**rule) for rule in rules_data.get("rules", [])]
            self._save_tag_rules()
        except Exception as e:
            logger.error(f"Failed to update tag rules: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame, column_mapping: ColumnMapping) -> pd.DataFrame:
        """データの前処理を実行"""
        try:
            # 必要な列の存在確認
            required_columns = [column_mapping.text_column]
            if column_mapping.id_column and column_mapping.id_column in df.columns:
                required_columns.append(column_mapping.id_column)
            if column_mapping.group_column and column_mapping.group_column in df.columns:
                required_columns.append(column_mapping.group_column)
            
            # データを選択
            processed_df = df[required_columns].copy()
            
            # テキスト列の前処理
            processed_df[column_mapping.text_column] = processed_df[column_mapping.text_column].fillna("")
            
            # 空のテキストを除外
            processed_df = processed_df[processed_df[column_mapping.text_column].str.strip() != ""]
            
            # インデックスをリセット
            processed_df = processed_df.reset_index(drop=True)
            
            logger.info(f"Preprocessed data: {len(processed_df)} rows")
            return processed_df
        
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
