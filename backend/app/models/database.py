from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/legal_doc_ai")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Company(Base):
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="company")
    chat_sessions = relationship("ChatSession", back_populates="company")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    s3_key = Column(String, unique=True)
    content_type = Column(String)
    file_size = Column(Integer)
    company_id = Column(Integer, ForeignKey("companies.id"))
    processed = Column(String, default="pending")  # pending, processing, completed, failed
    summary = Column(Text)
    doc_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Relationships
    company = relationship("Company", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text)
    chunk_index = Column(Integer)
    embedding_id = Column(String)  # ID in vector store
    doc_metadata = Column(JSON)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    session_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String)  # user, assistant
    content = Column(Text)
    context_documents = Column(JSON)  # List of document IDs used for context
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()