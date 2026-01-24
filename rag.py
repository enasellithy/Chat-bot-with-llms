import os
from typing import List
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from prompt_manager import PromptManager
from TextClean import TextCleaner
from DynamicSectionDetector import DynamicSectionDetector
import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

embedding_model = "all-MiniLM-L6-v2"
max_words = 50
overlap = 10
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

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
        self.emb_norm = self.model.encode(chunks, normalize_embeddings= True)
        return self.emb_norm.astype('float32')
    
    def transform(self, texts):
        return self.model.encode(texts, normalize_embeddings= True).astype('float32')
    
class Retriever:
    def __init__(self, emb_norm):
        dim = emb_norm.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb_norm)
    
    def search(self, query_vec, top_k: 3):
        query_vec = np.array(query_vec).astype('float32')
        if len(query_vec.shape) == 1:
            query_vec =  query_vec.reshape(1, -1)
        sims, idx = self.index.search(query_vec, top_k)
        return sims, idx

class RAG:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.cleaner = TextCleaner(keep_numbers=True)
        self.chunks = []
        self.retriever = None
        self.prompt_manager = PromptManager()
        self.section_detector = DynamicSectionDetector()
        
    def load_document(self):
        with open(self.data_path, 'r', encoding="utf-8") as f:
            text = f.read()
            self.section_detector.build_from_document(text)
            clean_text = self.cleaner.clean(text)
        return clean_text
    
    def retrieve_with_dynamic_filter(self, query: str, top_k:int = 3) -> List[str]:
        target_section = self.section_detector.detect_section_for_query(query)
        if not target_section:
            return self.retrieve(query,top_k)
        query_embedding = self.embedder.transform([query])
        sims, idx = self.retriever.search(query_embedding, top_k)
        filtered_chunks = []
        section_keywords = self.section_detector.section_keywords.get(target_section, [])
        for i in idx[0]:
            chunk = self.chunks[i]
            chunk_lower = chunk.lower()
            if len(filtered_chunks) >= top_k:
                break
            if len(filtered_chunks) > 3:
                return self.retrieve(query, top_k)
        return filtered_chunks[:top_k]
    
    def build(self):
        text = self.load_document()
        self.chunks = self.chunker.chunks(text)
        embeddings = self.embedder.fit_transform(self.chunks)
        self.retriever = Retriever(embeddings)
    
    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.transform([query])
        sims, idx = self.retriever.search(query_embedding, top_k)
        retrieved_chunks = [self.chunks[i] for i in idx[0]]
        return retrieved_chunks
    
    def generate(self, query: str, top_k: int = 5):
        retrieved_chunks = self.retrieve(query, top_k)
        context = "\n".join(retrieved_chunks)
        prompt = self.prompt_manager.compress_text(context, query)
        return prompt, context
    
    def process_query(self, query: str, top_k: int = 5, chat_history: List = None):
        cleaned_query = self.cleaner.clean(query)        
        retrieved_chunks = self.retrieve_with_dynamic_filter(query, top_k)
        context = "\n".join(retrieved_chunks)
        prompt = self.prompt_manager.compress_text(context, cleaned_query)
        target_section = self.section_detector.detect_section_for_query(query)
        if target_section:
            keywords = self.section_detector.section_keywords.get(target_section, [])
            print(f"[DEBUG] Section: {target_section}, Keywords: {keywords[:3]}")
        
        return {"query": query, "prompt": prompt, "context": context}