import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import os
import uuid

class VectorStore:
    def __init__(self):
        # Initialize ChromaDB
        # self.client = chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        # ))
        persist_path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False)
        )

        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            metadata={"description": "Legal document embeddings"}
        )

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store and return their IDs"""
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids

    async def similarity_search(self, query: str, k: int = 5, company_id: int = None, document_ids: List[int] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Build where filter supporting optional company and document filters
        def build_where(use_string_types: bool) -> Dict[str, Any]:
            where: Dict[str, Any] = {}
            if company_id is not None:
                where["company_id"] = str(company_id) if use_string_types else company_id
            if document_ids:
                # Chroma supports $in operator via {"field": {"$in": [...]}}
                converted_ids = [str(d) for d in document_ids] if use_string_types else document_ids
                where["document_id"] = {"$in": converted_ids}
            return where if where else None

        # Try filtering by company and/or document both as int and as string to avoid type-mismatch issues
        query_results = []
        try:
            res_int = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=build_where(use_string_types=False)
            )
            query_results.append(res_int)
        except Exception:
            pass
        try:
            res_str = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=build_where(use_string_types=True)
            )
            query_results.append(res_str)
        except Exception:
            pass
        
        # Fallback: no where filter if nothing found
        if not query_results:
            res_any = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=None
            )
            query_results.append(res_any)

        # Merge and format results
        search_results = []
        seen_ids = set()
        for results in query_results:
            if results and results.get('documents'):
                for i in range(len(results['documents'][0])):
                    item_id = results['ids'][0][i]
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)
                    item = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': 1 - results['distances'][0][i],
                        'id': item_id
                    }
                    # If filters were requested, enforce them post-query as well
                    if company_id is not None:
                        meta_company = item['metadata'].get('company_id')
                        if str(meta_company) != str(company_id):
                            continue
                    if document_ids:
                        meta_doc_id = item['metadata'].get('document_id')
                        if str(meta_doc_id) not in {str(i) for i in document_ids}:
                            continue
                    search_results.append(item)
        
        return search_results

    async def delete_document(self, document_id: int):
        """Delete all chunks for a document"""
        # Get all chunks for the document
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])

    async def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        # Try both int and string filters
        results = None
        try:
            results = self.collection.get(where={"document_id": document_id})
        except Exception:
            results = None
        if not results or not results.get('documents'):
            try:
                results = self.collection.get(where={"document_id": str(document_id)})
            except Exception:
                results = None
        
        chunks = []
        if results and results.get('documents'):
            for i in range(len(results['documents'])):
                chunks.append({
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'id': results['ids'][i]
                })
        
        return chunks