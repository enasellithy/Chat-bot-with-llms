from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import KeywordExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.ingestion import IngestionPipeline
from typing import List, Dict
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LlamaIndexRAG:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.index = None
        self.documents = []
        
        self.llm = self._setup_local_llm()
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5", 
            device="cpu"
        )
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                KeywordExtractor(keywords=5, llm=self.llm)
            ]
        )
    
    def _setup_local_llm(self):
        try:
            model_name = "distilgpt2" 
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None
            )
            
            llm = HuggingFaceLLM(
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                context_window=2048,
                generate_kwargs={"temperature": 0.1, "do_sample": True},
                device_map="auto" if device == "cuda" else None,
                model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {}
            )
            
            return llm
            
        except Exception as e:
            print(f"Warning: Could not load local LLM: {e}")
            print("Using a simpler approach without LLM for keyword extraction...")
            from llama_index.core.llms.mock import MockLLM
            return MockLLM()
    
    def prepare_dataset(self) -> List[Dict]:
        print("Reading files...")
        if self.data_dir and os.path.isdir(self.data_dir):
            self.documents = SimpleDirectoryReader(
                input_dir=self.data_dir,
                recursive=True,
                required_exts=['.txt', '.pdf', '.docx', '.md']
            ).load_data()
            print(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def build_index_with_cleaning(self):
        print("Cleaning and indexing documents...")
        if not self.documents:
            raise ValueError("No documents to index")
            
        # إذا كان KeywordExtractor بحاجة لـ LLM وليس لدينا واحد فعال،
        # نستخدم تحويلات أبسط
        if isinstance(Settings.llm, type) or hasattr(Settings.llm, '__class__') and Settings.llm.__class__.__name__ == 'MockLLM':
            cleaned_nodes = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=512, chunk_overlap=50),
                ]
            ).run(documents=self.documents)
        else:
            cleaned_nodes = self.pipeline.run(documents=self.documents)
            
        print("Building index...")
        self.index = VectorStoreIndex(
            nodes=cleaned_nodes,
            embed_model=self.embed_model
        )
        print("Index built successfully")
        return self.index
    
    def query_with_context(self, query: str, top_k: int = 5):
        if not self.index:
            raise ValueError("Index should be built first")
            
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact"
        )
        response = query_engine.query(query)
        
        return {
            "answer": str(response),
            "sources": self._extract_sources(response),
            "confidence": 0.8  
        }
    
    def _extract_sources(self, response):
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes[:3]:
                sources.append({
                    "text": node.text[:300] + "...",  
                    "score": float(node.score) if hasattr(node, 'score') else 0.0,
                    "metadata": node.metadata if hasattr(node, 'metadata') else {}
                })
        return sources