import torch
import torch.nn as nn
import torch.optim as optim
from torch import functional as F
from summarizer import Policy
from numba import cuda
import asyncio

from transformers import AutoModel, AutoTokenizer, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():    
    return "hello world"

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    input_text = data.get('text')
    input_text_tupled = (input_text,)

    summarizedText = policy.policy_inference(input_text_tupled)
    return jsonify({'summary': summarizedText})

@app.route('/summarize_better', methods=['POST'])
def summarize_better_text():
    data = request.get_json()
    input_text = data.get('text')
    input_text_tupled = (input_text,)

    summarizedText = policy_better.policy_inference(input_text_tupled)
    return jsonify({'summary': summarizedText})

if __name__ == '__main__':
    
    device = cuda.get_current_device()
    device.reset
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    BATCH_SIZE = 1
    EPOCH = 1
    LEARNING_RATE = 1e-4

    policy = Policy(_batch_size = BATCH_SIZE,
                _epoch= EPOCH,
                _lr = LEARNING_RATE,
                _model_version = "allenai/led-large-16384-arxiv",
                _device = device,
                # _model_version = "ver_1"
                )

    policy_better = Policy(_batch_size = BATCH_SIZE,
                _epoch= EPOCH,
                _lr = LEARNING_RATE,
                _model_version = "pszemraj/led-large-book-summary",
                _device = device,
                # _model_version = "ver_2"
                )

    app.run(host='0.0.0.0', port=8089)
    