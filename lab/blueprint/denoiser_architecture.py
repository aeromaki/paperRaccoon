import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")


class RoBERTa_Denoiser(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.roberta_encoder = AutoModel.from_config(AutoConfig.from_pretrained("xlm-roberta-large"))
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1024, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        hidden_state = self.roberta_encoder(input_ids, attention_mask).last_hidden_state

        output = torch.zeros((hidden_state.shape[0], 512, 1024)).to(self.device)
        output[:, :hidden_state.shape[1], :] = hidden_state

        output = self.head(output).squeeze(-1)
        
        return output


device = "cuda"
denoiser = RoBERTa_Denoiser(device).to(device)
denoiser.load_state_dict(torch.load("denoiser_roberta_rto_num_14000.pth"))


def denoise(text, max_seq_len=512):
    enc = tokenizer.encode(text)[1:-1]
    ll = len(enc)
    chunk = max_seq_len - 2

    ret = []
    for i in range(0, ll, chunk):
        input_ids = torch.tensor([[tokenizer.bos_token_id] + enc[i:i+chunk] + [tokenizer.eos_token_id]]).to(device)
        
        with torch.no_grad():
            remove = (denoiser(input_ids) > 0).to("cpu")
        
        for j, k in zip(input_ids[0,1:-1], remove[0,1:-1]):
            if not k:
                ret += [j]

    return tokenizer.decode(ret)