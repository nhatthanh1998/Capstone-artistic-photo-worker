import sys
import os
from dotenv import load_dotenv
from src.workers.generator import GeneratorWorker
import argparse
import requests
import json


QUEUE_HOST='amqps://nhatthanhlolo1:nhatthanh123@b-bb75efcd-b132-429f-9d91-9a062463a388.mq.ap-southeast-1.amazonaws.com:5671'
MAIN_SERVER='http://backendserverloadbalancer-1655295085.ap-southeast-1.elb.amazonaws.com'
# MAIN_SERVER='http://192.168.1.26:3001'

if __name__ == '__main__':
    try:
        snapshots = {}
        response = requests.get(f"{MAIN_SERVER}/styles/all-snapshots")
        data = json.loads(response.content.decode('utf-8'))
        for item in data:
            snapshots[item["id"]] = item["snapshotPath"]

        generator_worker = GeneratorWorker(
            queue_host=QUEUE_HOST,
            main_server_endpoint=MAIN_SERVER,
            snapshots=snapshots
        )
        generator_worker.start_task()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            sys.exit(0)
