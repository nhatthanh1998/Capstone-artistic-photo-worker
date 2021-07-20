import torch
from torchvision.transforms import transforms
from dotenv import load_dotenv
import os
import json
import boto3
import io
from torchvision.utils import make_grid
from PIL import Image

load_dotenv()
ENV = os.environ.get("ENV", "dev")
S3_BUCKET = os.environ.get("S3_BUCKET")

def init_s3_bucket(env, bucket):
    if env == "production":
        s3_client = boto3.client('s3')
    else:
        AWS_PUBLIC_KEY = os.environ.get("AWS_PUBLIC_KEY")
        AWS_PRIVATE_KEY = os.environ.get("AWS_PRIVATE_KEY")
        session = boto3.Session(
            aws_access_key_id=AWS_PUBLIC_KEY,
            aws_secret_access_key=AWS_PRIVATE_KEY
        )
        s3_client = session.client('s3')

    region = s3_client.get_bucket_location(Bucket=bucket)['LocationConstraint']

    return s3_client, region


s3, S3_REGION = init_s3_bucket(bucket=S3_BUCKET, env=ENV)


def transform():
    """ Transforms for training images """
    transform_ = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.ToTensor()
        ]
    )
    return transform_


def transform_tensor_to_bytes(tensor):
    grid = make_grid(tensor)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


def transform_byte_to_object(byte_data):
    response = byte_data.decode('utf8')
    response = json.loads(response)
    return response


def get_s3_location(key):
    return f'https://s3-{S3_REGION}.amazonaws.com/{S3_BUCKET}/{key}'


def save_image_to_s3(binary_data, key):
    s3.put_object(Body=binary_data, Bucket=S3_BUCKET, Key=key)
    return get_s3_location(key)
