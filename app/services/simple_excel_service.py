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
            # ビジネス関連のキーワード辞書（働き方、会社業績、仲間など）
            business_keywords = {
                "働き方": ["働き方", "ワークライフバランス", "残業", "休暇", "リモートワーク", "在宅勤務", "フレックス", "時短", "勤務時間", "労働環境"],
                "会社業績": ["業績", "売上", "利益", "成長", "拡大", "成功", "目標", "達成", "KPI", "業界", "市場", "競合"],
                "仲間・チーム": ["仲間", "チーム", "同僚", "上司", "部下", "協力", "連携", "コミュニケーション", "関係", "人間関係", "職場"],
                "キャリア": ["キャリア", "昇進", "昇格", "転職", "スキル", "経験", "学習", "成長", "挑戦", "目標", "将来"],
                "会社文化": ["文化", "風土", "価値観", "理念", "方針", "ルール", "慣習", "伝統", "雰囲気", "環境"],
                "給与・待遇": ["給与", "給料", "年収", "ボーナス", "賞与", "福利厚生", "待遇", "手当", "報酬", "インセンティブ"],
                "仕事内容": ["仕事", "業務", "プロジェクト", "タスク", "責任", "役割", "職務", "作業", "成果", "結果"],
                "会社評価": ["評価", "評判", "信頼", "信用", "ブランド", "イメージ", "知名度", "地位", "ポジション"],
                "満足度": ["満足", "良い", "素晴らしい", "優秀", "完璧", "最高", "気に入り", "おすすめ", "良い", "快適"],
                "不満・改善": ["不満", "悪い", "問題", "困る", "改善", "要望", "残念", "課題", "問題点", "改善点"]
            }
            
            # テキストからキーワードを抽出（より詳細な分析）
            all_words = []
            text_analysis = []
            
            for text in texts:
                if not text or text.strip() == '':
                    continue
                
                # 簡単な前処理
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend(words)
                
                # テキストの特徴を分析
                text_features = self._analyze_text_features(text)
                text_analysis.append(text_features)
            
            if not all_words:
                return []
            
            # 頻出単語をカウント
            word_counts = Counter(all_words)
            
            # テキスト分析結果を統合
            combined_analysis = self._combine_text_analysis(text_analysis)
            
            # ビジネスカテゴリベースのタグ候補を生成
            tag_candidates = []
            
            # データ分析結果に基づいてタグ候補を生成
            for category, keywords in business_keywords.items():
                category_score = 0
                category_count = 0
                
                # キーワードマッチング
                for keyword in keywords:
                    if keyword in word_counts:
                        category_score += word_counts[keyword]
                        category_count += word_counts[keyword]
                
                # データ分析結果からの補強
                if category in combined_analysis.get('business_summary', {}):
                    analysis_score = combined_analysis['business_summary'][category]
                    category_score += analysis_score * 2  # 分析結果を重み付け
                    category_count += analysis_score
                
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

    def _analyze_text_features(self, text: str) -> Dict[str, Any]:
        """テキストの特徴を分析"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentiment_indicators': {
                'positive': len(re.findall(r'良い|素晴らしい|最高|満足|気に入り|おすすめ', text)),
                'negative': len(re.findall(r'悪い|問題|困る|不満|残念|改善', text)),
                'neutral': 0
            },
            'business_indicators': {
                'work_style': len(re.findall(r'働き方|残業|休暇|リモート|在宅|フレックス', text)),
                'performance': len(re.findall(r'業績|売上|利益|成長|目標|達成', text)),
                'team': len(re.findall(r'仲間|チーム|同僚|上司|部下|協力', text)),
                'career': len(re.findall(r'キャリア|昇進|スキル|経験|学習|成長', text)),
                'culture': len(re.findall(r'文化|風土|価値観|理念|方針|雰囲気', text)),
                'compensation': len(re.findall(r'給与|給料|年収|ボーナス|待遇|報酬', text))
            }
        }
        return features

    def _combine_text_analysis(self, text_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """複数のテキスト分析結果を統合"""
        if not text_analyses:
            return {}
        
        combined = {
            'total_texts': len(text_analyses),
            'avg_length': sum(t['length'] for t in text_analyses) / len(text_analyses),
            'avg_word_count': sum(t['word_count'] for t in text_analyses) / len(text_analyses),
            'sentiment_summary': {
                'positive': sum(t['sentiment_indicators']['positive'] for t in text_analyses),
                'negative': sum(t['sentiment_indicators']['negative'] for t in text_analyses),
                'neutral': sum(t['sentiment_indicators']['neutral'] for t in text_analyses)
            },
            'business_summary': {
                'work_style': sum(t['business_indicators']['work_style'] for t in text_analyses),
                'performance': sum(t['business_indicators']['performance'] for t in text_analyses),
                'team': sum(t['business_indicators']['team'] for t in text_analyses),
                'career': sum(t['business_indicators']['career'] for t in text_analyses),
                'culture': sum(t['business_indicators']['culture'] for t in text_analyses),
                'compensation': sum(t['business_indicators']['compensation'] for t in text_analyses)
            }
        }
        return combined
    
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
