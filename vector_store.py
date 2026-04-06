import os
import pickle
from typing import List, Dict
from pathlib import Path
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import re

from semantic_chunking import SemanticChunkingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """LangChain FAISS vector store for semantic search on Confluence documents"""
    
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str, embedding_deployment: str):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=embedding_deployment,
            chunk_size=16  # For batch processing
        )
        self.chunking_service = SemanticChunkingService(self.embeddings)
        self.vectorstore = None
        self.documents = []
        self.chunk_report = []
    
    def create_index(
        self,
        documents: List[Dict],
        source_name: str = 'confluence',
        force_single_chunk: bool = False,
    ):
        """Create FAISS index from documents using LangChain"""
        logger.info("Creating FAISS index with LangChain...")

        langchain_docs, chunk_report = self.chunking_service.chunk_documents(
            documents,
            source_name=source_name,
            force_single_chunk=force_single_chunk,
        )
        self.chunk_report = chunk_report
        
        logger.info(f"Created {len(langchain_docs)} document chunks from {len(documents)} documents")
        
        # Create FAISS vector store
        logger.info("Generating embeddings and building FAISS index...")
        self.vectorstore = FAISS.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings
        )
        
        self.documents = documents
        logger.info(f"FAISS index created successfully")

    def save_chunk_report(self, report_path: str):
        """Save per-document chunking details to JSON"""
        self.chunking_service.save_chunk_report(self.chunk_report, report_path)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for most relevant chunks using LangChain FAISS"""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        # Use similarity search with score
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            result = {
                'text': doc.page_content,
                'title': doc.metadata.get('title'),
                'space': doc.metadata.get('space'),
                'type': doc.metadata.get('type'),
                'url': doc.metadata.get('url'),
                'distance': float(score),
                'relevance_score': 1 / (1 + float(score))
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_retriever(self, k: int = 5):
        """Get LangChain retriever for ConversationalRetrievalChain"""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return None
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def save_index(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata to disk"""
        index_dir = os.path.dirname(index_path)
        index_name = Path(index_path).stem

        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        self.vectorstore.save_local(index_dir, index_name=index_name)
        
        # Save metadata
        metadata = {
            'documents': self.documents,
            'chunk_report': self.chunk_report,
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata from disk"""
        index_dir = os.path.dirname(index_path)
        index_name = Path(index_path).stem
        index_pkl_path = os.path.join(index_dir, f"{index_name}.pkl")
        
        if (
            not os.path.exists(index_path)
            or not os.path.exists(index_pkl_path)
            or not os.path.exists(metadata_path)
        ):
            logger.error("Index or metadata files not found")
            return False
        
        # Load FAISS index
        self.vectorstore = FAISS.load_local(
            index_dir,
            self.embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.documents = metadata['documents']
        self.chunk_report = metadata.get('chunk_report', [])
        
        logger.info(f"Index loaded successfully")
        return True


def extract_github_urls(text: str) -> List[str]:
    """Extract GitHub repository URLs from text."""
    github_urls = []
    
    # Regular expression for matching full GitHub URLs
    full_url_pattern = r'https?://github\.com/[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+'
    
    # Regular expression for matching shorthand GitHub repo references like 'username/repo'
    shorthand_pattern = r'([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)'
    
    # Find all full URLs
    github_urls.extend(re.findall(full_url_pattern, text))
    
    # Find all shorthand repo references and convert them to full URLs
    shorthand_repos = re.findall(shorthand_pattern, text)
    for user, repo in shorthand_repos:
        github_urls.append(f'https://github.com/{user}/{repo}')
        logger.info(f"Extracted GitHub URLs: {github_urls}")
    
    return list(set(github_urls))
