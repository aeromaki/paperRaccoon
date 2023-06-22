import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class RewardModel(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

        self.led = AutoModel.from_config(AutoConfig.from_pretrained("allenai/led-large-16384-arxiv")).get_encoder()
        self.bart = AutoModel.from_config(AutoConfig.from_pretrained("facebook/bart-large")).get_encoder()

        self.flatten = nn.Linear(1024, 1)

        self.head = nn.Sequential(
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


device = "cuda"
model = RewardModel(device).to(device)
model.load_state_dict(torch.load("reward_model_openai_2190.pth"))