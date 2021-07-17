import torch
from PIL import Image
from src.models.generator import Generator
import requests
from src.utils.utils import load_model, transform, transform_byte_to_object, save_image_to_s3, transform_tensor_to_bytes
import uuid
import pika
from datetime import datetime


class GeneratorWorker:
    def __init__(self, queue_host,
                snapshots,
                 main_server_endpoint):
        self.device = "cpu"
        self.main_server_endpoint = main_server_endpoint
        self.connection = pika.BlockingConnection(pika.URLParameters(queue_host))
        self.channel = self.connection.channel()
        self.snapshots = snapshots
        # self.snapshots = {
        #     '8b634cf4-112f-46ce-ac23-ea00f772c152': 'https://artisan-model-snapshots.s3.amazonaws.com/candy/1626159002212?AWSAccessKeyId=AKIAYB7AFYKUNGXO6WOW&Expires=1626619955&Signature=SvjrOJyJVpYmGdaz0WA4Et1UQwg%3D',
        #     '70c82d96-1cbb-4090-84d3-2b1721af088f': 'https://artisan-model-snapshots.s3.amazonaws.com/rain_princess/1626352835484?AWSAccessKeyId=AKIAYB7AFYKUNGXO6WOW&Expires=1626619955&Signature=VNL3OiP8KFGRJNzzvKDEiyPR0w0%3D',
        #     'f96ab7b3-fbe3-48dd-a60a-8e729fb80621': 'https://artisan-model-snapshots.s3.amazonaws.com/udnie/1626355468951?AWSAccessKeyId=AKIAYB7AFYKUNGXO6WOW&Expires=1626619955&Signature=TFMrz6yVXk%2FLRmfpt%2BvHNLvO1vk%3D',
        #     '64b4ef33-e1ce-473b-9d4d-62356b9091e1': 'https://artisan-model-snapshots.s3.amazonaws.com/new_style/1626527641075?AWSAccessKeyId=AKIAYB7AFYKUNGXO6WOW&Expires=1626619955&Signature=ViQFTJyVcT6dCH7N7Hlt5GQNmG0%3D'
        # }
        self.weights = {}
        for key, value in self.snapshots.items():
            self.weights[key] = torch.hub.load_state_dict_from_url(value, map_location=torch.device(self.device))
        self.generator = Generator().to(self.device)
        self.transform_ = transform()
    
    def load_weight(self, style_id):
        self.generator.load_state_dict(self.weights[style_id])

    def upload_model(self, snapshot_location):
        self.generator = load_model(path=snapshot_location, generator=self.generator, device=self.device)

    def preprocess(self, style_id, photo_access_url):
        self.load_weight(style_id)
        model_input = Image.open(requests.get(photo_access_url, stream=True).raw)
        model_input = self.transform_(model_input).unsqueeze(0)
        return model_input.to(self.device)

    def inference(self, model_input):
        return self.generator(model_input)[0]

    def post_process(self, model_output, image_name, userId, style_id):
        byte_data = transform_tensor_to_bytes(model_output)
        image_location = save_image_to_s3(byte_data, image_name)
        endpoint_url = f"{self.main_server_endpoint}/medias/transfer-photo/completed"
        payload = {'userId': userId, 'transferPhotoLocation': image_location, 'styleId': style_id}
        requests.post(endpoint_url, data=payload)
        torch.cuda.empty_cache()

    def handler(self, ch, method, photo_access_url, userId, image_name, style_id):
        # 1. Preprocess
        model_input = self.preprocess(style_id=style_id, photo_access_url=photo_access_url)

        # 2. Transform
        model_output = self.inference(model_input=model_input)

        # 3. Post process
        self.post_process(model_output=model_output, image_name=image_name, userId=userId, style_id=style_id)

        # 4. Ack the processed message.
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_transfer_photo_task(self, ch, method, properties, body):
        print("Transfer photo task on process...")
        data = transform_byte_to_object(body)
        style_id = data['styleId']
        # extract data from body
        userId = data['userId']
        accessURL = data['accessURL']
        date_time = datetime.now().strftime("%m-%d-%Y")
        image_name = f"{date_time}/{uuid.uuid4()}.jpg"
        # Put data to model process pipeline
        self.handler(ch=ch, method=method, photo_access_url=accessURL, userId=userId, image_name=image_name,
                     style_id=style_id)
        print("Transfer done")

    def process_update_model_task(self, ch, method, properties, body):
        print("Start update model....")
        body = transform_byte_to_object(body)
        data = body['data']
        snapshot_location = data['snapshotLocation']
        self.upload_model(snapshot_location)

    def init_transfer_photo_queue(self):
        self.channel.queue_declare("TRANSFER_PHOTO_QUEUE", durable=True)
        self.channel.exchange_declare("TRANSFER_PHOTO_EXCHANGE", exchange_type='direct')
        self.channel.queue_bind(exchange="TRANSFER_PHOTO_EXCHANGE", queue="TRANSFER_PHOTO_QUEUE", routing_key="")
        self.channel.basic_consume(queue="TRANSFER_PHOTO_QUEUE", on_message_callback=self.process_transfer_photo_task)
        print(f' [*] Waiting for messages at TRANSFER PHOTO EXCHANGE. To exit press CTRL+C')

    def declare_update_model_workflow(self):
        rs = self.channel.queue_declare(queue='', exclusive=True)
        queue_name = rs.method.queue
        self.channel.exchange_declare(exchange="UPDATE_WEIGHT_EXCHANGE", exchange_type='fanout')
        self.channel.queue_bind(exchange="UPDATE_WEIGHT_EXCHANGE", queue=queue_name)
        self.channel.basic_consume(queue=queue_name, on_message_callback=self.process_update_model_task)

    def start_task(self):
        self.init_transfer_photo_queue()
        self.declare_update_model_workflow()
        self.channel.start_consuming()
