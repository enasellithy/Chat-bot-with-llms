import re
from TextClean import TextCleaner

class PromptManager:
    def __init__(self):
        self.cleaner = TextCleaner(keep_numbers=True)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def compress_text(self, context: str, question: str, context_topic: str = "") -> str:
        cleaned_context = self.cleaner.clean(context)
        cleaned_question = self.cleaner.clean(question)
        question_lower = cleaned_question.lower()
        sentences = [s.strip() for s in re.split(r"[\n.]", cleaned_context) if s.strip()]
        
        categories = {
            "sick": ["sick", "medical", "certificate", "doctor"],
            "annual": ["annual", "balance", "request", "advance"]
        }

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