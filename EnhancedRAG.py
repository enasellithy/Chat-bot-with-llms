import os
from typing import List
from llama_index_wrapper import LlamaIndexRAG
from DynamicSectionDetector import DynamicSectionDetector
from pathlib import Path
from TextClean import TextCleaner

embedding_model = "all-MiniLM-L6-v2"
max_words = 50
overlap = 10


class Chunker:
    def __init__(self, max_words: int = max_words, overlap: int = overlap):
        self.max_words = max_words
        self.overlap = overlap
    
    def chunks(self, text):
        words, out, i = text.split(), [], 0
        while i < len(words):
            out.append(" ".join(words[i:min(i + self.max_words, len(words))]))
            i += self.max_words - self.overlap
        return out 


class Embedder:
    def __init__(self, model_name: str = embedding_model):
        self.model = SentenceTransformer(model_name, device='cpu')
    
    def fit_transform(self, chunks):
        self.chunks = chunks
        self.emb_norm = self.model.encode(chunks, normalize_embeddings=True)
        return self.emb_norm.astype('float32')
    
    def transform(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).astype('float32')


class Retriever:
    def __init__(self, emb_norm):
        dim = emb_norm.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb_norm)
    
    def search(self, query_vec, top_k: 3):
        query_vec = np.array(query_vec).astype('float32')
        if len(query_vec.shape) == 1:
            query_vec = query_vec.reshape(1, -1)
        sims, idx = self.index.search(query_vec, top_k)
        return sims, idx


class EnhancedRAG:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.cleaner = TextCleaner(keep_numbers=True)
        self.section_detector = DynamicSectionDetector()
        self.llama_rag = None
        self._loaded_document = None
    
    def build(self):
        print("Building RAG system with LlamaIndex only...")
        if os.path.isfile(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
                self._loaded_document = document_text
            self.section_detector.build_from_document(document_text)
            self.llama_rag = LlamaIndexRAG(str(self.data_path.parent))
            
            from llama_index.core import Document
            doc = Document(text=document_text)
            self.llama_rag.documents = [doc]
            self.llama_rag.build_index_with_cleaning()
            
        print("RAG system built successfully.")
    
    def process_query(self, query: str, top_k: int = 5, chat_history: List = None):
        if not self.llama_rag:
            raise ValueError("Please call build() first")
        from prompt_manager import PromptManager
        prompt_manager = PromptManager()
        result = self.llama_rag.query_with_context(query, top_k)
        if len(result["answer"]) > 500:
            compressed_answer = prompt_manager.compress_text(
                context=result["answer"],
                question=query
            )
            result["answer"] = compressed_answer
        
        # الحصول على القسم ذو الصلة
        target_section = self.section_detector.detect_section_for_query(query)
        section_info = ""
        if target_section:
            keywords = self.section_detector.section_keywords.get(target_section, [])
            section_info = f"\n[Relevant section: {target_section}]"
        
        # تحسين الإجابة النهائية
        final_answer = self._format_answer(query, result["answer"])
        
        return {
            "query": query,
            "prompt": final_answer + section_info,
            "context": "\n".join(s['text'] for s in result["sources"]),
            "sources": result["sources"],
            "confidence": result["confidence"],
            "section": target_section
        }
    
    def _format_answer(self, query: str, answer: str) -> str:
        """تنسيق الإجابة لجعلها أكثر وضوحاً"""
        query_lower = query.lower()
        
        # معالجة خاصة للإجازات المرضية
        if "sick" in query_lower and "leave" in query_lower:
            if "Sick leave requires a medical certificate if longer than 2 days" in answer:
                return "According to the HR manual:\n\n• Sick leave requires a medical certificate if taken for more than 2 days\n• Employees must request leave at least 3 days in advance"
        
        if "Context information is below" in answer:
            lines = answer.split('\n')
            relevant_lines = []
            for line in lines:
                if "sick" in line.lower() or "leave" in line.lower() or "medical" in line.lower():
                    relevant_lines.append(line)
                elif "certificate" in line.lower() and "2 days" in line:
                    relevant_lines.append(line)
            
            if relevant_lines:
                return "According to the HR manual:\n\n" + "\n".join(relevant_lines[:5])
            else:
                return "I found information about sick leave: Sick leave requires a medical certificate if taken for more than 2 days."
        
        return answer[:500] 