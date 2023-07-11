from pydantic import BaseModel
from transformers import LEDModel, LEDTokenizer, BartModel
# from datasets import load_dataset
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def generate_global_attention_mask(tokenizer, input_ids):
    mask = torch.zeros_like(input_ids)
    mask[((input_ids == tokenizer.bos_token_id) | (input_ids == tokenizer.eos_token_id)).nonzero(as_tuple=True)] = 1
    return mask

# Reward Model
class RewardModel(nn.Module):
    def __init__(self, _device, device="cuda"):
        super(RewardModel, self).__init__()

        self.device = _device

        self.led = LEDModel.from_pretrained("allenai/led-large-16384-arxiv").get_encoder()
        self.bart = BartModel.from_pretrained("facebook/bart-large").get_encoder()
        self.tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

        self.flatten = nn.Linear(1024, 1)

        self.head = nn.Sequential (
            nn.Linear(17408, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, summary_input_ids, global_attention_mask=None):
        hidden_state = self.led(input_ids, global_attention_mask=global_attention_mask).last_hidden_state
        output = torch.zeros((hidden_state.shape[0], 16384, 1024)).to(self.device) # head가 fixed size만 받을 수 있으므로 0으로 padding
        output[:, :hidden_state.shape[1], :] = hidden_state

        bart_hidden_state = self.bart(summary_input_ids).last_hidden_state
        bart_output = torch.zeros((bart_hidden_state.shape[0], 1024, 1024)).to(self.device) # head가 fixed size만 받을 수 있으므로 0으로 padding
        bart_output[:, :bart_hidden_state.shape[1], :] = bart_hidden_state

        concat = torch.cat([output.repeat((summary_input_ids.shape[0], 1, 1)), bart_output], dim=1)
        concat = self.flatten(concat)
        result = self.head(concat.transpose(1, 2))

        return result.squeeze()

    def reward_inference(self, data):
        # Prepare the inputs
        device = self.device
        input_ids = self.tokenizer.encode(data["text"], return_tensors="pt").to(device)
        summary_input_ids = self.tokenizer.encode(data["summary"], return_tensors="pt").to(device)
        att = generate_global_attention_mask(self.tokenizer, input_ids).to(device)
        
        print("Shape of input_ids: ", input_ids.shape) 
        print("Shape of summary_input_ids: ", summary_input_ids.shape)
        print("Shape of att: ", att.shape)

        # Inference
        with torch.no_grad():
            output = self.forward(input_ids, summary_input_ids, att)
        return output.tolist()

class InferenceInput(BaseModel):
    text: str
    summary: str


