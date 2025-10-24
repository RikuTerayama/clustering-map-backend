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
            # より具体的なビジネスキーワード辞書
            business_keywords = {
                "残業問題": ["残業", "22時", "夜", "遅く", "長時間労働", "過労", "深夜", "夜勤", "時間外"],
                "ワークライフバランス": ["ワークライフバランス", "プライベート", "家族", "休暇", "休み", "余暇", "生活", "時間"],
                "連絡・コミュニケーション": ["連絡", "電話", "メール", "夜", "休日", "緊急", "呼び出し", "連絡先"],
                "チームワーク": ["チーム", "仲間", "同僚", "協力", "助け合い", "連携", "サポート", "支え"],
                "上司・部下関係": ["上司", "部下", "マネージャー", "リーダー", "管理", "指導", "評価", "フィードバック"],
                "キャリア成長": ["スキル", "スキルアップ", "研修", "学習", "成長", "経験", "知識", "能力向上"],
                "昇進・昇格": ["昇進", "昇格", "昇給", "ポジション", "役職", "責任", "権限", "地位"],
                "給与・待遇": ["給与", "給料", "年収", "ボーナス", "賞与", "手当", "福利厚生", "待遇"],
                "会社業績": ["業績", "売上", "利益", "成長", "目標", "達成", "成功", "拡大"],
                "会社文化": ["文化", "風土", "価値観", "理念", "方針", "ルール", "慣習", "雰囲気"],
                "仕事内容": ["プロジェクト", "タスク", "業務", "作業", "責任", "役割", "成果", "結果"],
                "職場環境": ["環境", "オフィス", "設備", "スペース", "快適", "使いやすい", "整備", "改善"],
                "満足・不満": ["満足", "不満", "良い", "悪い", "問題", "改善", "要望", "期待", "希望"]
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
                    # カテゴリ全体のスコア（適切なカテゴリのみ）
                    if self._is_valid_tag(category):
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
                    # 不適切なタグをフィルタリング
                    if self._is_valid_tag(word):
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
                'positive': len(re.findall(r'良い|素晴らしい|最高|満足|気に入り|おすすめ|快適|嬉しい', text)),
                'negative': len(re.findall(r'悪い|問題|困る|不満|残念|改善|禁止|保てない|難しい', text)),
                'neutral': 0
            },
            'business_indicators': {
                '残業問題': len(re.findall(r'残業|22時|夜|遅く|長時間労働|過労|深夜|夜勤|時間外', text)),
                'ワークライフバランス': len(re.findall(r'ワークライフバランス|プライベート|家族|休暇|休み|余暇|生活|時間', text)),
                '連絡・コミュニケーション': len(re.findall(r'連絡|電話|メール|夜|休日|緊急|呼び出し|連絡先', text)),
                'チームワーク': len(re.findall(r'チーム|仲間|同僚|協力|助け合い|連携|サポート|支え', text)),
                '上司・部下関係': len(re.findall(r'上司|部下|マネージャー|リーダー|管理|指導|評価|フィードバック', text)),
                'キャリア成長': len(re.findall(r'スキル|スキルアップ|研修|学習|成長|経験|知識|能力向上', text)),
                '昇進・昇格': len(re.findall(r'昇進|昇格|昇給|ポジション|役職|責任|権限|地位', text)),
                '給与・待遇': len(re.findall(r'給与|給料|年収|ボーナス|賞与|手当|福利厚生|待遇', text)),
                '会社業績': len(re.findall(r'業績|売上|利益|成長|目標|達成|成功|拡大', text)),
                '会社文化': len(re.findall(r'文化|風土|価値観|理念|方針|ルール|慣習|雰囲気', text)),
                '仕事内容': len(re.findall(r'プロジェクト|タスク|業務|作業|責任|役割|成果|結果', text)),
                '職場環境': len(re.findall(r'環境|オフィス|設備|スペース|快適|使いやすい|整備|改善', text))
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
                '残業問題': sum(t['business_indicators']['残業問題'] for t in text_analyses),
                'ワークライフバランス': sum(t['business_indicators']['ワークライフバランス'] for t in text_analyses),
                '連絡・コミュニケーション': sum(t['business_indicators']['連絡・コミュニケーション'] for t in text_analyses),
                'チームワーク': sum(t['business_indicators']['チームワーク'] for t in text_analyses),
                '上司・部下関係': sum(t['business_indicators']['上司・部下関係'] for t in text_analyses),
                'キャリア成長': sum(t['business_indicators']['キャリア成長'] for t in text_analyses),
                '昇進・昇格': sum(t['business_indicators']['昇進・昇格'] for t in text_analyses),
                '給与・待遇': sum(t['business_indicators']['給与・待遇'] for t in text_analyses),
                '会社業績': sum(t['business_indicators']['会社業績'] for t in text_analyses),
                '会社文化': sum(t['business_indicators']['会社文化'] for t in text_analyses),
                '仕事内容': sum(t['business_indicators']['仕事内容'] for t in text_analyses),
                '職場環境': sum(t['business_indicators']['職場環境'] for t in text_analyses)
            }
        }
        return combined

    def _is_valid_tag(self, tag: str) -> bool:
        """タグが適切かどうかを判定"""
        # 長すぎるタグを除外（10文字以上）
        if len(tag) > 10:
            return False
        
        # 不適切なパターンを除外
        invalid_patterns = [
            r'^[0-9]+$',  # 数字のみ
            r'^[a-zA-Z]+$',  # 英字のみ
            r'です$',  # 敬語で終わる
            r'ます$',  # 敬語で終わる
            r'ください$',  # 依頼で終わる
            r'いただきたい$',  # 依頼で終わる
            r'ですが$',  # 逆接で終わる
            r'ので$',  # 理由で終わる
            r'が$',  # 助詞で終わる
            r'を$',  # 助詞で終わる
            r'に$',  # 助詞で終わる
            r'で$',  # 助詞で終わる
            r'と$',  # 助詞で終わる
            r'から$',  # 助詞で終わる
            r'まで$',  # 助詞で終わる
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, tag):
                return False
        
        return True
    
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
