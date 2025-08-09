import os
import asyncio
from rq import Queue
from redis import Redis

from app.services.document_processor import DocumentProcessor


def get_redis_connection() -> Redis:
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    return Redis(host=redis_host, port=redis_port, db=redis_db)


def get_queue(name: str = "default") -> Queue:
    return Queue(name, connection=get_redis_connection())


def process_document_job(document_id: int) -> None:
    """Synchronous RQ job entrypoint that runs the async processor.

    RQ executes sync callables. We create an event loop to run the async
    processing pipeline.
    """
    processor = DocumentProcessor()
    asyncio.run(processor.process_document(document_id))


