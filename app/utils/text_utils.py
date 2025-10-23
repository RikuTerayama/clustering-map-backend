import re
import unicodedata
from typing import List, Set, Dict, Any
import logging
from sudachipy import tokenizer
from sudachipy import dictionary

logger = logging.getLogger(__name__)

# SudachiPyの設定
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

# ストップワード（基本的なもの）
STOP_WORDS = {
    'の', 'に', 'は', 'を', 'が', 'で', 'と', 'も', 'から', 'まで', 'より', 'へ', 'や', 'か', 'など',
    'こと', 'もの', 'ため', 'とき', 'ところ', 'よう', 'そう', 'これ', 'それ', 'あれ', 'どれ',
    'この', 'その', 'あの', 'どの', 'ここ', 'そこ', 'あそこ', 'どこ', 'だ', 'である', 'です',
    'ます', 'でした', 'でした', 'です', 'だ', 'である', 'です', 'ます', 'です', 'ます',
    'です', 'ます', 'です', 'ます', 'です', 'ます', 'です', 'ます', 'です', 'ます'
}


def normalize_text(text: str) -> str:
    """テキストを正規化"""
    if not isinstance(text, str):
        return ""
    
    # Unicode正規化（NFKC）
    text = unicodedata.normalize('NFKC', text)
    
    # 全角半角統一
    text = text.replace('　', ' ')  # 全角スペースを半角に
    
    # 連続する空白を単一の空白に
    text = re.sub(r'\s+', ' ', text)
    
    # 前後の空白を削除
    text = text.strip()
    
    return text


def remove_special_characters(text: str) -> str:
    """特殊文字を除去"""
    # 英数字、ひらがな、カタカナ、漢字、基本的な記号のみ残す
    text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', ' ', text)
    
    # 連続する空白を単一の空白に
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize_japanese(text: str) -> List[str]:
    """日本語テキストをトークン化"""
    try:
        tokens = []
        for token in tokenizer_obj.tokenize(text, mode):
            # 品詞情報を取得
            pos = token.part_of_speech()
            
            # 名詞、動詞、形容詞のみを抽出
            if pos[0] in ['名詞', '動詞', '形容詞']:
                # 活用形を基本形に変換
                surface = token.surface()
                if surface and len(surface) > 1:  # 1文字のトークンは除外
                    tokens.append(surface)
        
        return tokens
    except Exception as e:
        logger.warning(f"Tokenization failed: {e}")
        # フォールバック: 単純な分割
        return text.split()


def remove_stop_words(tokens: List[str]) -> List[str]:
    """ストップワードを除去"""
    return [token for token in tokens if token not in STOP_WORDS]


def preprocess_text(text: str) -> List[str]:
    """テキストの前処理を実行"""
    # 正規化
    text = normalize_text(text)
    
    # 特殊文字除去
    text = remove_special_characters(text)
    
    # トークン化
    tokens = tokenize_japanese(text)
    
    # ストップワード除去
    tokens = remove_stop_words(tokens)
    
    return tokens


def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """テキストからキーワードを抽出"""
    tokens = preprocess_text(text)
    
    # 頻度をカウント
    from collections import Counter
    token_counts = Counter(tokens)
    
    # 頻度順にソートして上位を返す
    return [token for token, count in token_counts.most_common(max_keywords)]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """2つのテキストの類似度を計算（Jaccard係数）"""
    tokens1 = set(preprocess_text(text1))
    tokens2 = set(preprocess_text(text2))
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union if union > 0 else 0.0


def merge_similar_tags(tags: List[str], threshold: float = 0.8) -> List[str]:
    """類似したタグをマージ"""
    if not tags:
        return []
    
    merged_tags = []
    used_indices = set()
    
    for i, tag1 in enumerate(tags):
        if i in used_indices:
            continue
            
        similar_tags = [tag1]
        used_indices.add(i)
        
        for j, tag2 in enumerate(tags[i+1:], i+1):
            if j in used_indices:
                continue
                
            similarity = calculate_text_similarity(tag1, tag2)
            if similarity >= threshold:
                similar_tags.append(tag2)
                used_indices.add(j)
        
        # 最も長いタグを代表とする
        representative = max(similar_tags, key=len)
        merged_tags.append(representative)
    
    return merged_tags
