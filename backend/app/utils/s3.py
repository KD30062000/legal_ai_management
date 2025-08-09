import boto3
import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

def generate_presigned_url(file_name: str, content_type: str):
    s3 = boto3.client("s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("S3_REGION")
    )

    bucket = os.getenv("S3_BUCKET_NAME")
    key = f"uploads/{uuid4()}_{file_name}"

    url = s3.generate_presigned_url('put_object', 
        Params={
            'Bucket': bucket,
            'Key': key,
            'ContentType': content_type
        },
        ExpiresIn=300  # expires in 5 minutes
    )

    return {"upload_url": url, "s3_key": key}
