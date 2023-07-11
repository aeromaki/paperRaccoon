import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch import functional as F
from summarizer import Policy
from numba import cuda
import asyncio
from tqdm import tqdm
from rewardModel import RewardModel
from pydantic import BaseModel

from transformers import AutoModel, AutoTokenizer, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, LEDModel, LEDTokenizer, BartModel

from flask import Flask, request, jsonify
app = Flask(__name__)

class InferenceInput(BaseModel):
    text: str
    summary: str

@app.route('/rm', methods=['GET'])
def index():    
    return "hello world for rm"
    
@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    input_data = InferenceInput(text=data["text"], summary=data["summary"])
    rewards = reward_model.reward_inference(data)

    return {"result": rewards}

if __name__ == '__main__':
    
    device = cuda.get_current_device()
    device.reset
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    reward_model = RewardModel(device).to(device)
    reward_model.load_state_dict(torch.load("reward_model_openai_2190.pth"))
    reward_model.eval()

    app.run(host='0.0.0.0', port=8089)
    