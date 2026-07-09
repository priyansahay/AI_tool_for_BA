from pathlib import Path
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.custom_exception import log_exception

try:
    EMBEDDING_MODEL_NAME = ("all-MiniLM-L6-v2")
    vector_db = "vector_db"
    INDEX_PATH = (f"{vector_db}/faiss_index.bin")
    DOCUMENT_PATH = (f"{vector_db}/documents.pkl")

    def load_embedding_model():
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return model
    
    def create_documents(report_text,forecast_metrics,churn_metrics,anomaly_summary,segment_summary):
        documents = [
            f"Executive Report: {report_text}",
            f"Forecast Metrics: {forecast_metrics}",
            f"churn Metrics: {churn_metrics}",
            f"Anomaly Summary: {anomaly_summary}",
            f"Segment Summary: {segment_summary}"
        ]
        return documents
    
    def create_embeddings(documents,embedding_model):
        embeddings = (embedding_model.encode(documents, convert_to_numpy = True))
        return embeddings
    
    def create_faiss_index(embeddings):
        dimension = (embeddings.shape[1])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    
    def save_vector_store(index, documents):
        Path(vector_db).mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, INDEX_PATH)
        with open(DOCUMENT_PATH, "wb") as file:
            pickle.dump(documents, file)

    def load_vector_store():
        index = faiss.read_index(INDEX_PATH)
        with open(DOCUMENT_PATH, "rb") as file:
            documents = pickle.load(file)
        return index, documents
    
    def retrive_context(query, embedding_model, index, documents, top_k=3):
        query_embedding = (embedding_model.encode([query], convert_to_numpy = True))
        distance, indices = (index.search(query_embedding, top_k))
        contexts = []
        for idx in indices[0]:                    
            contexts.append(documents[idx])
        return contexts
    
    def build_context(retrieved_docs):
        context = "\n\n".join(retrieved_docs)
        return context
    
    def rag_ingestion_pipeline(report_text, forecast_metrics, churn_metrics, anomaly_summary, segment_summary):
        embedding_model = (load_embedding_model())
        documents = (create_documents(report_text, forecast_metrics,churn_metrics, anomaly_summary, segment_summary))
        embeddings = (create_embeddings(documents, embedding_model))
        index = (create_faiss_index(embeddings))
        save_vector_store(index, documents)
        return index, embedding_model
except Exception as e:
    log_exception(e)
    raise