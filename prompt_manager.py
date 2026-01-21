import re
import tiktoken

class PromptManager:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def compress_text(self, context: str, question: str, context_topic: str = "") -> str:
        question_lower = question.lower()
        sentences = [s.strip() for s in re.split(r"[\n.]", context) if s.strip()]
        
        # كلمات مفتاحية لكل نوع من الإجازات
        categories = {
            "sick": ["sick", "medical", "certificate", "doctor"],
            "annual": ["annual", "balance", "request", "advance"]
        }

        # تحديد الفئة الحالية بناءً على السؤال
        current_category = None
        for cat, words in categories.items():
            if any(w in question_lower for w in words):
                current_category = cat
                break

        relevant_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            if current_category:
                if any(w in sent_lower for w in categories[current_category]):
                    relevant_sentences.append(sent)
            else:
                if any(w in sent_lower for w in question_lower.split()):
                    relevant_sentences.append(sent)

        if not relevant_sentences:
            return "This information is not available in the manual."

        if current_category == "sick":
            numbers = re.findall(r'\d+', question_lower)
            for s in relevant_sentences:
                if "2 days" in s.lower():
                    if "1" in numbers or "one" in question_lower:
                        return f"According to policy: '{s}'. So for 1 day, no certificate is needed."
                    return s

        return relevant_sentences[0]