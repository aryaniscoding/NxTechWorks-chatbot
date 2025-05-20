from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def chunk_and_embed(full_text: str):
    texts = splitter.split_text(full_text)
    embeddings = model.encode(texts,batch_size = 64,show_progress_bar = True ,convert_to_tensor=False)
    return texts, embeddings
