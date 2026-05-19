import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from config import PATH_TO_STORAGE, COLLECTION_NAME, EMBED_MODEL_NAME, LLM_NAME
from llama_index.core import PromptTemplate

# Modèles
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)
Settings.llm = Ollama(model=LLM_NAME)

def load_index():
    client = chromadb.PersistentClient(path=PATH_TO_STORAGE)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)

def ask(question: str) -> dict:
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=3,response_mode="compact")
    response = query_engine.query(question)
    
    qa_prompt = PromptTemplate(
        "Tu es un assistant juridique expert en droit du travail français. "
        "Réponds toujours en français.\n" 
        "Contexte :\n{context_str}\n"
        "Question : {query_str}\n"
        "Réponse :"
    )
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
    response = query_engine.query(question)
    
    return {
        "answer": str(response),
        "sources": [node.text[:200] for node in response.source_nodes]
    }

if __name__ == "__main__":
    result = ask("Est-ce qye les entreprises de travail temporaire peuvent exercer l'activité d'entreprise de travail à temps partagé ?")
    print(result["answer"])
    print("\nSources :")
    for s in result["sources"]:
        print(s)