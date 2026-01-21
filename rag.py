import os
from typing import List
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
from TextClean import TextCleaner
# from DatasetPrep import DatasetPrParation 
from prompt_manager import PromptManager
embedding_model = "all-MiniLM-L6-v2"
max_words = 50
overlap = 10
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        self.chunks = []
        self.retriever = None
        self.prompt_manager = PromptManager()
        
    def load_document(self):
        with open(self.data_path, 'r', encoding="utf-8") as f:
            return f.read()
    
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
        current_query_words = set(query.lower().split())
        sick_words = {"sick", "medical", "doctor"}
        annual_words = {"annual", "vacation", "balance"}
        
        context_topic = ""
        if chat_history:
            last_user_query = chat_history[-1]["user"].lower()
            if any(w in query.lower() for w in sick_words | annual_words):
                 context_topic = "" 
            else:
                 context_topic = last_user_query

        search_query = f"{context_topic} {query}".strip()
        retrieved_chunks = self.retrieve(search_query, top_k)
        context = "\n".join(retrieved_chunks)
        
        prompt = self.prompt_manager.compress_text(context, query, context_topic)
        return {"query": query, "prompt": prompt, "context": context}