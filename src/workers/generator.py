import torch
from PIL import Image
from src.models.generator import Generator
import requests
from src.utils.utils import transform, transform_byte_to_object, save_image_to_s3, transform_tensor_to_bytes
import uuid
import pika
from datetime import datetime


class GeneratorWorker:
    def __init__(self, queue_host,
                snapshots,
                 main_server_endpoint):
        self.device = "cpu"
        self.main_server_endpoint = main_server_endpoint
        self.queue_host = queue_host
        self.connection = None
        self.channel = None
        self.weights = {}
        for style_id, snapshot_path in snapshots.items():
            self.update_weight(style_id=style_id, snapshot_path=snapshot_path)
        self.generator = Generator().to(self.device)
        self.transform_ = transform()
    
    def load_weight(self, style_id):
        self.generator.load_state_dict(self.weights[style_id])

    def update_weight(self, style_id, snapshot_path):
        self.weights[style_id] = torch.hub.load_state_dict_from_url(snapshot_path, map_location=torch.device(self.device))

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
        model_input = self.preprocess(style_id=style_id, photo_access_url=photo_access_url)
        model_output = self.inference(model_input=model_input)
        self.post_process(model_output=model_output, image_name=image_name, userId=userId, style_id=style_id)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_transfer_photo_task(self, ch, method, properties, body):
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

    def handle_update_weight(self, ch, method, properties, body):
        print("Start update weight library...")
        body = transform_byte_to_object(body)
        style_id = body['styleId']
        snapshot_path = body['snapshotPath']
        self.update_weight(style_id=style_id, snapshot_path=snapshot_path)
        print("Update weights completed")
        

    def init_transfer_photo_queue(self):
        self.channel.queue_declare("TRANSFER_PHOTO_QUEUE", durable=True)
        self.channel.exchange_declare("TRANSFER_PHOTO_EXCHANGE", exchange_type='direct')
        self.channel.queue_bind(exchange="TRANSFER_PHOTO_EXCHANGE", queue="TRANSFER_PHOTO_QUEUE", routing_key="")
        self.channel.basic_consume(queue="TRANSFER_PHOTO_QUEUE", on_message_callback=self.process_transfer_photo_task)
        print(f' [*] Waiting for messages at TRANSFER PHOTO EXCHANGE. To exit press CTRL+C')

    def init_update_weight_queue(self):
        rs = self.channel.queue_declare(queue='', exclusive=True)
        queue_name = rs.method.queue
        self.channel.exchange_declare(exchange="UPDATE_WEIGHT_EXCHANGE", exchange_type='fanout')
        self.channel.queue_bind(exchange="UPDATE_WEIGHT_EXCHANGE", queue=queue_name)
        self.channel.basic_consume(queue=queue_name, on_message_callback=self.handle_update_weight)

    def start_task(self):
        i = 0
        while True:
            try:
                print("Connecting...")
                self.connection = pika.BlockingConnection(pika.URLParameters(self.queue_host))
                self.channel = self.connection.channel()
                self.init_transfer_photo_queue()
                self.init_update_weight_queue()
                self.channel.basic_qos(prefetch_count=1)
                self.channel.start_consuming()
            except KeyboardInterrupt:
                self.channel.stop_consuming()
                self.connection.close()
                break
            except pika.exceptions.ConnectionClosedByBroker:
                continue
            except pika.exceptions.AMQPChannelError as err:
                print(err)
                continue
            except pika.exceptions.AMQPConnectionError as err:
                print(err)
                print("Connection was closed, retrying...")
                break;
