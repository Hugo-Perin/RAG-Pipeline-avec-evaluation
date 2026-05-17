from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# 0. Choisir le modèle
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="gemma3:4b")

# 1. Charger les docs
path_to_xml_folder = "data"
documents = SimpleDirectoryReader(path_to_xml_folder).load_data()

# 2. Découpage des docs
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64) #chaque node a 512 tokens, avec un chevauchement de 64 tokens (pour ne pas couper phrase et préserver contexte)
nodes = splitter.get_nodes_from_documents(documents)

# 3. Embeddings + stockage dans ChromaDB
client = chromadb.PersistentClient(path="./storage") #client car il passe commande au serveur chromadb, ./storage est l'endroit de l'entrepot
collection = client.get_or_create_collection("rag_docs") #dans l'entrepot ./storage, on crée une partie spécifique "rag_docs"

#
vector_store = ChromaVectorStore(chroma_collection=collection) #ChromaVectorStore fait l'intermédiaire entre ChromaDB et LlamaIndex
storage_context = StorageContext.from_defaults(vector_store=vector_store) #manuel de procédure pour LlamaIndex

index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True) #"bouton démarrer" qui exécute 