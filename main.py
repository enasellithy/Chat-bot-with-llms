import os
import sys
from rag import RAG
from EnhancedRAG import EnhancedRAG
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



if __name__ == "__main__":
    # rag = RAG(data_path="hr_manual.txt")
    rag = EnhancedRAG(data_path="hr_manual.txt")
    rag.build()
    print("Downloaded and built RAG system successfully.")
    
    chat_history = []
    
    while True:
        query = input("Enter your question: ")
        if query.lower() in ['hi', 'hello', 'مرحبا', 'أهلاً', 'سلام', 'سلام عليكم', 'السلام عليكم']:
            print(f"👋 {query.lower()}")
            continue
        elif query.lower() in ['thx', 'thanks', 'شكراً', 'شكرا', 'جزاك الله خيراً', 'جزاك الله خير']:
            print(f"👋 {query.lower()} any thing els: ")
            continue
        elif query.lower() in ['خروج', 'exit', 'quit',"goodbye","bye","مع السلامة"]:
            print(f"👋 {query.lower()} Thank for use bot")
            break
        else:
        
            result = rag.process_query(query, chat_history=chat_history)
            chat_history.append({"user": query, "agent": result['prompt']})
            print(f"\n🤖 Agent: {result['prompt']}\n")
