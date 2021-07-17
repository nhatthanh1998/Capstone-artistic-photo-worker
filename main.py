import sys
import os
from dotenv import load_dotenv
from src.workers.generator import GeneratorWorker
import argparse
import requests
import json


load_dotenv()
QUEUE_HOST = os.environ.get("QUEUE_HOST")
MAIN_SERVER_ENDPOINT = os.environ.get("MAIN_SERVER_ENDPOINT")

if __name__ == '__main__':
    try:
        snapshots = {}
        response = requests.get(f"{MAIN_SERVER_ENDPOINT}/styles/all-snapshots")
        data = json.loads(response.content.decode('utf-8'))
        for item in data:
            snapshots[item["id"]] = item["snapshotPath"]
        
        generator_worker = GeneratorWorker(
            queue_host=QUEUE_HOST,
            main_server_endpoint=MAIN_SERVER_ENDPOINT,
            snapshots=snapshots
        )
        generator_worker.start_task()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            sys.exit(0)
