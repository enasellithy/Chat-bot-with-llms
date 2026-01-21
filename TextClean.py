import re
import unicodedata
from typing import Optional

class TextCleaner:    
    def __init__(self, keep_numbers: bool = False, custom_stopwords: Optional[list] = None):
        self.keep_numbers = keep_numbers
        
        self.base_stopwords = {
            'ar': ['في', 'من', 'على', 'إلى', 'أن', 'هذا', 'هذه', 'كان', 'هل', 'أو'],
            'en': ['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'is', 'are']
        }
        
        self.custom_stopwords = custom_stopwords or []
    
    def detect_language(self, text: str) -> str:
        arabic_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        if arabic_count > len(text) * 0.3:
            return 'ar'
        else:
            return 'en' 
    
    def normalize_unicode(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([char for char in text if not unicodedata.combining(char)])
        return text
    
    def remove_special_chars(self, text: str, lang: str) -> str:
        """إزالة الرموز الخاصة والحفاظ على ما يناسب اللغة [citation:1][citation:10]."""
        if lang == 'ar':
            pattern = r'[^\u0600-\u06FF\s]'
            if self.keep_numbers:
                pattern = r'[^\u0600-\u06FF0-9\s]'
        else: 
            pattern = r'[^a-zA-Z\s]'
            if self.keep_numbers:
                pattern = r'[^a-zA-Z0-9\s]'
        
        text = re.sub(pattern, ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean(self, text: str, lang: Optional[str] = None) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        if not lang:
            lang = self.detect_language(text)
        
        if lang == 'en':
            text = text.lower()
        
        text = self.normalize_unicode(text)
        
        text = self.remove_special_chars(text, lang)
        
        words = text.split()
        stopwords_to_remove = self.base_stopwords.get(lang, []) + self.custom_stopwords
        words = [word for word in words if word not in stopwords_to_remove]
        
        clean_text = ' '.join(words)
        
        return clean_text