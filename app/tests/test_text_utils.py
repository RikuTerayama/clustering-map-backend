import pytest
from app.utils.text_utils import (
    normalize_text, remove_special_characters, tokenize_japanese,
    remove_stop_words, preprocess_text, extract_keywords_from_text,
    calculate_text_similarity, merge_similar_tags
)


class TestTextUtils:
    """テキスト処理ユーティリティのテスト"""
    
    def test_normalize_text(self):
        """テキスト正規化のテスト"""
        # 全角半角統一
        assert normalize_text("　テスト　") == "テスト"
        assert normalize_text("テスト  テスト") == "テスト テスト"
        
        # 空文字列
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
    
    def test_remove_special_characters(self):
        """特殊文字除去のテスト"""
        text = "テスト！@#$%^&*()_+{}|:<>?[]\\;'\",./"
        result = remove_special_characters(text)
        assert "！" not in result
        assert "@" not in result
        assert "テスト" in result
    
    def test_tokenize_japanese(self):
        """日本語トークン化のテスト"""
        text = "顧客満足度を向上させたい"
        tokens = tokenize_japanese(text)
        assert len(tokens) > 0
        assert isinstance(tokens, list)
    
    def test_remove_stop_words(self):
        """ストップワード除去のテスト"""
        tokens = ["顧客", "の", "満足度", "を", "向上", "させたい"]
        result = remove_stop_words(tokens)
        assert "の" not in result
        assert "を" not in result
        assert "顧客" in result
        assert "満足度" in result
    
    def test_preprocess_text(self):
        """テキスト前処理のテスト"""
        text = "顧客満足度を向上させたい！"
        tokens = preprocess_text(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_extract_keywords_from_text(self):
        """キーワード抽出のテスト"""
        text = "顧客満足度を向上させたい。お客様との関係性を深めることが重要だと思う。"
        keywords = extract_keywords_from_text(text, max_keywords=5)
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
    
    def test_calculate_text_similarity(self):
        """テキスト類似度計算のテスト"""
        text1 = "顧客満足度を向上させたい"
        text2 = "顧客満足度を高めたい"
        similarity = calculate_text_similarity(text1, text2)
        assert 0 <= similarity <= 1
        
        # 同じテキスト
        same_similarity = calculate_text_similarity(text1, text1)
        assert same_similarity == 1.0
        
        # 全く異なるテキスト
        different_similarity = calculate_text_similarity(text1, "全く異なる内容")
        assert different_similarity < 0.5
    
    def test_merge_similar_tags(self):
        """類似タグマージのテスト"""
        tags = ["顧客満足度", "顧客満足", "満足度向上", "システム品質", "品質向上"]
        merged = merge_similar_tags(tags, threshold=0.8)
        assert len(merged) <= len(tags)
        assert isinstance(merged, list)
    
    def test_empty_input_handling(self):
        """空入力の処理テスト"""
        assert preprocess_text("") == []
        assert extract_keywords_from_text("") == []
        assert calculate_text_similarity("", "") == 1.0
        assert merge_similar_tags([]) == []
