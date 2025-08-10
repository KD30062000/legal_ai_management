# from fastapi import APIRouter, UploadFile
# from app.utils.s3 import generate_presigned_url

# router = APIRouter()

# @router.post("/upload/")
# async def upload_document(file: UploadFile):
#     result = generate_presigned_url(file.filename, file.content_type)
#     return result

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.utils.s3 import generate_presigned_url
from app.models.database import get_db, Company, Document as DBDocument
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.queues.jobs import get_queue, process_document_job
from typing import Optional
import boto3
import os
from uuid import uuid4

router = APIRouter()
document_processor = DocumentProcessor()
vector_store = VectorStore()
llm_service = LLMService()
PROCESS_ON_POST = os.getenv("PROCESS_ON_POST", "true").lower() == "true"


class UploadCompleteRequest(BaseModel):
    document_id: int

# @router.post("/upload/")
# async def upload_document(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     company_name: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     """Upload document and start processing"""
    
#     # Get or create company
#     normalized_name = (company_name or "").strip()
#     company = (
#         db.query(Company)
#         .filter(func.lower(Company.name) == func.lower(normalized_name))
#         .first()
#     )
#     if not company:
#         company = Company(name=normalized_name)
#         db.add(company)
#         db.commit()
#         db.refresh(company)
    
#     # Generate presigned URL for upload
#     upload_info = generate_presigned_url(file.filename, file.content_type)
    
#     # Create document record
#     document = DBDocument(
#         filename=file.filename,
#         s3_key=upload_info["s3_key"],
#         content_type=file.content_type,
#         file_size=0,  # Will be updated after upload
#         company_id=company.id,
#         processed="pending"
#     )
#     db.add(document)
#     db.commit()
#     db.refresh(document)
    
#     # Add background task to process document after upload (optional)
#     if PROCESS_ON_POST:
#         # Enqueue to Redis queue for processing
#         q = get_queue()
#         q.enqueue(process_document_job, document.id)
    
#     return {
#         "document_id": document.id,
#         "upload_url": upload_info["upload_url"],
#         "s3_key": upload_info["s3_key"],
#         "message": "Document uploaded successfully. Processing will begin shortly."
#     }


@router.post("/upload/direct")
async def upload_document_direct(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company_name: str = Form(...),
    db: Session = Depends(get_db)
):
    """Single-call upload: backend uploads to S3 and (optionally) triggers processing."""

    # Validate content type
    allowed_content_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Get or create company
    normalized_name = (company_name or "").strip()
    company = (
        db.query(Company)
        .filter(func.lower(Company.name) == func.lower(normalized_name))
        .first()
    )
    if not company:
        company = Company(name=normalized_name)
        db.add(company)
        db.commit()
        db.refresh(company)

    # Upload directly to S3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("S3_REGION"),
    )
    bucket = os.getenv("S3_BUCKET_NAME")
    s3_key = f"uploads/{uuid4()}_{file.filename}"
    file_bytes = await file.read()
    s3.put_object(Bucket=bucket, Key=s3_key, Body=file_bytes, ContentType=file.content_type)

    # Create document record
    document = DBDocument(
        filename=file.filename,
        s3_key=s3_key,
        content_type=file.content_type,
        file_size=len(file_bytes),
        company_id=company.id,
        processed="pending",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Queue processing
    q = get_queue()
    q.enqueue(process_document_job, document.id)

    return {
        "document_id": document.id,
        "s3_key": s3_key,
        "message": "Upload successful. Processing has been queued.",
    }

async def process_document_after_upload(document_id: int, s3_key: str):
    """Background task to process document after upload"""
    try:
        # Wait a bit to ensure upload is complete
        import asyncio
        await asyncio.sleep(5)
        
        # Start processing
        await document_processor.process_document(document_id)
    except Exception as e:
        print(f"Error processing document {document_id}: {str(e)}")


# @router.post("/upload/complete")
# async def upload_complete(
#     request: UploadCompleteRequest,
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db)
# ):
#     """Trigger processing after the client successfully PUT the file to S3.

#     Call this endpoint immediately after the S3 upload responds 200.
#     """
#     document = db.query(DBDocument).filter(DBDocument.id == request.document_id).first()
#     if not document:
#         raise HTTPException(status_code=404, detail="Document not found")

#     # Verify the object exists in S3 before starting
#     try:
#         document_processor.s3_client.head_object(
#             Bucket=os.getenv("S3_BUCKET_NAME"),
#             Key=document.s3_key,
#         )
#     except Exception:
#         raise HTTPException(status_code=400, detail="S3 object not found yet. Ensure the PUT succeeded.")

#     q = get_queue()
#     q.enqueue(process_document_job, document.id)
#     return {"status": "queued", "document_id": document.id}

@router.get("/documents/{company_name}")
async def get_company_documents(
    company_name: str,
    db: Session = Depends(get_db)
):
    """Get all documents for a company"""
    company = db.query(Company).filter(Company.name == company_name).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    documents = db.query(DBDocument).filter(DBDocument.company_id == company.id).all()
    
    return {
        "company": company.name,
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "processed": doc.processed,
                "summary": doc.summary,
                "created_at": doc.created_at,
                "processed_at": doc.processed_at
            }
            for doc in documents
        ]
    }

@router.get("/documents/{document_id}/status")
async def get_document_status(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get processing status of a document"""
    document = db.query(DBDocument).filter(DBDocument.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document.id,
        "filename": document.filename,
        "processed": document.processed,
        "created_at": document.created_at,
        "processed_at": document.processed_at,
        "summary": document.summary,
        "metadata": document.doc_metadata
    }


@router.get("/documents/{document_id}/structured-summary")
async def get_document_structured_summary(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Generate and return a structured summary for a single document.

    Uses the existing vector store chunks as context and the LLM's
    structured summarization prompt. Does not persist the result; returns it.
    """
    document = db.query(DBDocument).filter(DBDocument.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.processed != "completed":
        raise HTTPException(status_code=400, detail="Document not processed yet")

    # Retrieve all chunks for the document from the vector store
    chunks = await vector_store.get_document_chunks(document_id)
    if not chunks:
        return {
            "document_id": document_id,
            "summary": "No chunks found for this document. Try reprocessing the upload.",
            "document_count": 0,
            "generated_at": "now"
        }

    result = await llm_service.generate_structured_summary(chunks)
    result.update({"document_id": document_id})
    return result
