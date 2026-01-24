# dynamic_metadata.py

import re
from typing import List, Dict, Set
from collections import Counter

class DynamicSectionDetector:
    
    def __init__(self, min_keyword_freq: int = 2):
        self.min_freq = min_keyword_freq
        self.section_keywords = {}
        self.section_titles = []
    
    def extract_sections_from_text(self, text: str) -> Dict[str, List[str]]:
        sections = {}
        
        lines = text.split('\n')
        current_section = "General"
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            if re.match(r'^\d+[\.\)]?\s+', line) or line in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.']:
                if current_content and current_section != "General":
                    section_text = ' '.join(current_content)
                    keywords = self._extract_keywords_for_section(section_text)
                    sections[current_section] = keywords
                
                current_section = line
                current_content = []
            elif line and not line.startswith('-'):
                current_content.append(line)
        
        if current_content and current_section != "General":
            section_text = ' '.join(current_content)
            keywords = self._extract_keywords_for_section(section_text)
            sections[current_section] = keywords
        
        return sections
    
    def _extract_keywords_for_section(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        stop_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'can', 'may', 'might', 'must', 'shall', 'employee', 'employees',
            'hr', 'management', 'process', 'system', 'company', 'required'
        }
        words = text.split()
        filtered_words = []
        for word in words:
            if (len(word) >= 4 and  
                word not in stop_words and
                not word.isdigit()):
                filtered_words.append(word)
        word_freq = Counter(filtered_words)
        keywords = []
        for word, freq in word_freq.most_common(20):  
            if freq >= self.min_freq and self._is_good_keyword(word):
                keywords.append(word)
        
        return keywords[:10] 
    
    def _is_good_keyword(self, word: str) -> bool:
        general_words = {
            'policy', 'policies', 'procedure', 'procedures', 'manual',
            'document', 'documents', 'section', 'chapter', 'part'
        }
        
        if word in general_words:
            return False
        
        if len(word) < 4:
            return False
        
        return True
    
    def build_from_document(self, document_text: str):
        """بناء قاعدة الكلمات المفتاحية من المستند"""
        sections = self.extract_sections_from_text(document_text)
        
        # تحويل الأسماء الطويلة إلى أسماء مختصرة
        for full_title, keywords in sections.items():
            short_name = self._get_short_section_name(full_title)
            self.section_keywords[short_name] = keywords
            self.section_titles.append(full_title)
        
        return self.section_keywords
    
    def _get_short_section_name(self, full_title: str) -> str:
        cleaned = re.sub(r'^\d+[\.\)]?\s*', '', full_title).strip()
        
        if 'leave' in cleaned.lower():
            return 'leave'
        elif 'payroll' in cleaned.lower():
            return 'payroll'
        elif 'attendance' in cleaned.lower():
            return 'attendance'
        elif 'recruitment' in cleaned.lower() or 'hiring' in cleaned.lower():
            return 'hiring'
        elif 'onboarding' in cleaned.lower():
            return 'onboarding'
        elif 'performance' in cleaned.lower():
            return 'performance'
        elif 'offboarding' in cleaned.lower() or 'exit' in cleaned.lower():
            return 'offboarding'
        elif 'records' in cleaned.lower():
            return 'records'
        else:
            first_word = cleaned.split()[0].lower()
            return first_word if len(first_word) > 3 else 'general'
    
    def detect_section_for_query(self, query: str) -> str:
        query_lower = query.lower()
        
        best_match = None
        best_score = 0
        
        for section_name, keywords in self.section_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = section_name
        
        if best_score >= 1:  
            return best_match
        
        return None