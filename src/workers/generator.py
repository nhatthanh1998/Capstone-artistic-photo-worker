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
                 exchange_transfer_photo_name,
                 exchange_update_model_name,
                 routing_key,
                 snapshot_path,
                 main_server_endpoint):
        self.device = "cpu"
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.snapshot_path = snapshot_path
        self.queue_host = queue_host
        self.exchange_transfer_photo_name = exchange_transfer_photo_name
        self.exchange_update_model_name = exchange_update_model_name
        self.routing_key = routing_key
        self.main_server_endpoint = main_server_endpoint
        self.generator = Generator().to(self.device)
        self.transform_ = transform()
        self.generator = load_model(path=self.snapshot_path, generator=self.generator, device=self.device)
        self.connection = pika.BlockingConnection(pika.URLParameters(self.queue_host))
        self.channel = self.connection.channel()

    def upload_model(self, snapshot_location):
        self.generator = load_model(path=snapshot_location, generator=self.generator, device=self.device)

    def preprocess(self, photo_access_url):
        model_input = Image.open(requests.get(photo_access_url, stream=True).raw)
        model_input = self.transform_(model_input).unsqueeze(0)
        print(model_input.shape)
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
        model_input = self.preprocess(photo_access_url=photo_access_url)

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

    def declare_transfer_photo_workflow(self):
        self.channel.queue_declare(self.routing_key, durable=True)
        self.channel.exchange_declare(exchange=self.exchange_transfer_photo_name, exchange_type='direct')
        self.channel.queue_bind(exchange=self.exchange_transfer_photo_name, queue=self.routing_key,
                                routing_key=self.routing_key)
        self.channel.basic_consume(queue=self.routing_key, on_message_callback=self.process_transfer_photo_task)
        print(f' [*] Waiting for messages at exchange {self.exchange_transfer_photo_name} routing Key: {self.routing_key}. To exit press CTRL+C')

    def declare_update_model_workflow(self):
        rs = self.channel.queue_declare(queue='', exclusive=True)
        queue_name = rs.method.queue
        self.channel.exchange_declare(exchange=self.exchange_update_model_name, exchange_type='fanout')
        self.channel.queue_bind(exchange=self.exchange_update_model_name, queue=queue_name, routing_key=self.routing_key)
        self.channel.basic_consume(queue=queue_name, on_message_callback=self.process_update_model_task)

    def start_task(self):
        self.declare_transfer_photo_workflow()
        self.declare_update_model_workflow()
        self.channel.start_consuming()
