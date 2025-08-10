from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from typing import List, Optional
from app.models.database import (
    get_db, Company, ChatSession, ChatMessage, 
    Document as DBDocument
)
from app.models.chat_models import ChatRequest, ChatResponse, QuickPrompt
from datetime import datetime
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
import json
import time
from itertools import islice

router = APIRouter()
vector_store = VectorStore()
llm_service = LLMService()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """Chat with RAG using company documents"""
    start_time = time.time()
    
    # Get company (case-insensitive, trimmed)
    normalized_name = (request.company_name or "").strip()
    company = (
        db.query(Company)
        .filter(func.lower(Company.name) == func.lower(normalized_name))
        .first()
    )
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    # Get or create chat session
    if request.session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == request.session_id,
            ChatSession.company_id == company.id
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        session = ChatSession(
            company_id=company.id,
            session_name=request.session_name or f"Chat {len(company.chat_sessions) + 1}"
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    
    # Search for relevant documents with optional restriction to specific document IDs
    search_results = await vector_store.similarity_search(
        query=request.message,
        k=5,
        company_id=company.id,
        document_ids=request.specific_documents
    )

    # Fallback: if a specific document was requested but retrieval returned nothing,
    # load chunks for that document directly to ensure some context is provided.
    if (not search_results) and request.specific_documents:
        aggregated = []
        # Limit total chunks to avoid extremely long prompts
        max_total_chunks = 6
        for doc_id in request.specific_documents:
            chunks = await vector_store.get_document_chunks(doc_id)
            for chunk in islice(chunks, 0, max_total_chunks - len(aggregated)):
                # Ensure same shape used by similarity_search
                aggregated.append({
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'score': 1.0,
                    'id': chunk['id']
                })
            if len(aggregated) >= max_total_chunks:
                break
        search_results = aggregated

    # Decide which document IDs to associate with this exchange for history filtering
    result_doc_ids = [doc['metadata'].get('document_id') for doc in search_results]
    effective_doc_ids = request.specific_documents or result_doc_ids
    
    # Get recent chat history
    recent_messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session.id
    ).order_by(ChatMessage.timestamp.desc()).limit(10).all()
    
    chat_history = [
        {"role": msg.role, "content": msg.content}
        for msg in reversed(recent_messages)
    ]
    
    # Generate response using RAG
    response = await llm_service.generate_rag_response(
        query=request.message,
        context_documents=search_results,
        chat_history=chat_history
    )
    
    # Store user message
    user_message = ChatMessage(
        session_id=session.id,
        role="user",
        content=request.message,
        context_documents=effective_doc_ids
    )
    db.add(user_message)
    
    # Store assistant response
    assistant_message = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=response,
        context_documents=effective_doc_ids
    )
    db.add(assistant_message)
    
    # Touch session updated_at
    session.updated_at = datetime.utcnow()
    db.commit()
    
    processing_time = (time.time() - start_time) * 1000
    
    return ChatResponse(
        session_id=session.id,
        response=response,
        context_documents=[
            {
                "filename": doc['metadata'].get('filename'),
                "document_id": doc['metadata'].get('document_id'),
                "score": doc['score']
            }
            for doc in search_results
        ],
        processing_time_ms=processing_time
    )

@router.post("/chat/quick-prompt", response_model=ChatResponse)
async def quick_prompt(
    request: QuickPrompt,
    db: Session = Depends(get_db)
):
    """Handle predefined quick prompts"""
    start_time = time.time()
    
    # Define quick prompts
    prompt_templates = {
        "summarize": "Please provide a comprehensive summary of all the legal documents, focusing on the main purpose, key parties, and critical terms of each document.",
        
        "analyze_risks": "Analyze all the legal documents and identify potential risks, liabilities, and areas of concern that need attention. Highlight any clauses that could be problematic.",
        
        "key_terms": "Extract and explain all the key terms and conditions from the legal documents. Focus on the most important clauses, definitions, and provisions.",
        
        "obligations": "List all the obligations, duties, and responsibilities for each party mentioned in the legal documents. Organize them by party and document.",
        
        "deadlines": "Identify all important dates, deadlines, renewal dates, expiration dates, and time-sensitive requirements mentioned in the legal documents."
    }
    
    if request.prompt_type not in prompt_templates:
        raise HTTPException(status_code=400, detail="Invalid prompt type")
    
    # Convert to regular chat request
    chat_request = ChatRequest(
        company_name=request.company_name,
        message=prompt_templates[request.prompt_type],
        session_id=request.session_id,
        session_name=f"Quick Analysis - {request.prompt_type.title()}",
        specific_documents=request.specific_documents
    )
    
    return await chat_with_documents(chat_request, db)

@router.get("/chat/sessions/{company_name}")
async def get_chat_sessions(
    company_name: str,
    db: Session = Depends(get_db)
):
    """Get all chat sessions for a company"""
    normalized_name = (company_name or "").strip()
    company = (
        db.query(Company)
        .filter(func.lower(Company.name) == func.lower(normalized_name))
        .first()
    )
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    sessions = db.query(ChatSession).filter(
        ChatSession.company_id == company.id
    ).order_by(ChatSession.updated_at.desc()).all()
    
    return {
        "sessions": [
            {
                "id": session.id,
                "name": session.session_name,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages)
            }
            for session in sessions
        ]
    }

@router.get("/chat/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get all messages in a chat session"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.timestamp.asc()).all()
    
    return {
        "session_id": session_id,
        "session_name": session.session_name,
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "context_documents": msg.context_documents
            }
            for msg in messages
        ]
    }

@router.delete("/chat/sessions/{session_id}")
async def delete_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Delete a chat session and all its messages"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete all messages first
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    
    # Delete session
    db.delete(session)
    db.commit()
    
    return {"message": "Session deleted successfully"}