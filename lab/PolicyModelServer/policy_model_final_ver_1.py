import pandas as pd
import datetime
import time
# Model Save & Load
import os
# GPU Reset
from numba import cuda

import torch
import torch.nn as nn
import torch.optim as optim

from torch import functional as F
# from torchsummary import summary

# 모델, Tokenizer Load
from transformers import AutoModel, AutoTokenizer, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# 데이터셋 Load from Summarize_from_feedvback, Huggingface
from datasets import load_dataset

"""#### GPU Reset & Setting device"""

os.chdir('/content/drive/MyDrive/23_Conference')

def GPU_reset():
    device = cuda.get_current_device()
    device.reset
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    return device

device = GPU_reset()


BATCH_SIZE = 1
EPOCH = 1
LEARNING_RATE = 1e-4

class Data_Preprocessing:
    def __init__(self):
        # DownLoad Data from huggingFace
        ## Text Summary 데이터
        ## CNN, TL;DR, Daily Mail
        self.data_feedback = load_dataset("openai/summarize_from_feedback", 'axis')

        # Split into Train and Validation dataset
        # Convert to DataFrame
        self.df_train = pd.DataFrame(self.data_feedback['validation'])
        self.df_valid = pd.DataFrame(self.data_feedback['test'])


    # Original Text + Summarized Text 데이터 Columm 추출
    def Data_cleaning(self, df_train, df_valid):
        df_train['original_text'] = [row['post'] for row in df_train['info']]
        df_valid['original_text'] = [row['article'] for row in df_valid['info']]

        df_train['sum_text'] = [row['text'] for row in df_train['summary']]
        df_valid['sum_text'] = [row['text'] for row in df_valid['summary']]

        df_all = pd.concat([df_train[['original_text', 'sum_text']], df_valid[['original_text', 'sum_text']]], ignore_index=True)

        return df_all

    # 최종 DataFrame 출력
    def data_complete_form(self):
        df_train = self.Data_cleaning(self.df_train, self.df_valid).iloc[:-300, :]
        df_valid = df_train.iloc[-300: , :].reset_index(drop=True)

        return df_train, df_valid

# df_train.head(2)

"""### [2.2] DataLoader
- 원본 Text와 Summarize가 합쳐진 데이터 형식의 DataFrame을 DataLoader로 처리
- 입력: DataFrame <br>
- Feature : original_with_good_sum,   original_with_bad_sum  </br>
- 내용: 원본 텍스트 + 긍정 Summary, 원본 텍스트 + 부정 Summary
"""

class Policy_Dataset(torch.utils.data.Dataset):

    def __init__(self, df_textsum): #, transforms_=None, random_masking = False,  unaligned=True ):

        self.original_text = df_textsum['original_text']
        self.sum_text = df_textsum['sum_text']

        print(f"My_dataset __init__ received : {self.original_text.shape}, {self.sum_text.shape}")
        print(f"Data Type : {type(self.original_text[0]), type(self.sum_text[0])}")
        # print(f"Data example : {self.original_text[0], {self.sum_text[0]}}")

    def __getitem__(self, index):
        original_text = self.original_text[index]
        sum_text = self.sum_text[index]

        return original_text, sum_text


    def __len__(self):
        return len(self.sum_text)

"""##  [3] All In ONe
- Data Load 부터 Training, Inferecnce까지 담은 Class
"""

class Policy(Data_Preprocessing, Policy_Dataset):
    def __init__(self,
                 _batch_size = BATCH_SIZE,
                 _epoch= EPOCH,
                 _lr = LEARNING_RATE,
                 _model_name = "allenai/led-large-16384-arxiv",
                 _device = device):

        # Device
        self.device = _device

        # Data
        ## WARNING :: ONLY FOR FINETUNING
        self.train_loader, self.valid_loader = self.finetuning_data_load(_batch_size)

        # Tokenizer
        self.tokenizer = self.policy_tokenizer(model_name = _model_name)

        # Model
        self.policy_model = self.policy_model(model_name = _model_name)

        # Training
        self.epoch = _epoch
        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=_lr)

        print(f"\n===================== POLICY INIT COMPLETE =====================\n")


    ''' ========================================= D A T A  L O A D E R ===================================================='''
    # WARNING :: ONLY FOR FINETUNING
    def finetuning_data_load(self, batch_size):
      df_train, df_valid = Data_Preprocessing().data_complete_form()
      train_loader = torch.utils.data.DataLoader(Policy_Dataset(df_train), batch_size=batch_size, shuffle=False, drop_last = False)
      valid_loader = torch.utils.data.DataLoader(Policy_Dataset(df_valid), batch_size=batch_size, shuffle=False, drop_last = False)

      return train_loader, valid_loader


    ''' ========================================= T O K E N I Z E R ===================================================='''

    def policy_tokenizer(self, model_name = "allenai/led-large-16384-arxiv"):
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      return tokenizer

    ''' ================================== G L O B A L  A T T E N T I O N  M A S K ======================================'''

    def generate_global_attention_mask(self, tokenizer, input_ids):
      mask = torch.torch.zeros_like(input_ids)
      mask[((input_ids == tokenizer.bos_token_id) | (input_ids == tokenizer.eos_token_id)).nonzero(as_tuple=True)] = 1

      return mask


    ''' ========================================= 모  델  정  의 ===================================================='''

    def policy_model(self, model_name="allenai/led-large-16384-arxiv"):
      policy_model = LEDForConditionalGeneration.from_pretrained(model_name).to(self.device)

      return policy_model

    ''' ========================================= 모  델  저  장 ===================================================='''

    def save_model_info(self, _model, _version="ver_1"):
        if not os.path.isdir("./policy_model"):
            os.makedirs("./policy_model")
        # 모델 정보 저장
        _model = _model.cpu()
        torch.save({'model_state_dict': _model.state_dict(),
                    # 'optimizer_state_dict': _optimizer.state_dict(),
                    # 'record_list' : {'train_loss': _train_loss, 'valid_loss': _valid_loss},
                    }, f"./policy_model/policy_model_{_version}.pth")  #policy_model_ver_1

        print(f"******************* Model Saved : policy_model_{_version} *******************")


    ''' ========================================== T R A I N I N G  ====================================================='''
    # 모델 Training
    def train(self):

        ## 초기화
        train_loss_list = []
        valid_loss_list = []

        record_train_loss = []
        record_valid_loss = []

        # Optimizer & Loss function
        optimizer = self.optimizer

        # Data Loader
        train_loader = self.train_loader

        # 모델 정의
        model = self.policy_model
        tokenizer = self.tokenizer
        generate_global_attention_mask = self.generate_global_attention_mask

        # Hyper Parameter
        epoch = self.epoch
        device = self.device

        model.train()
        for i in range(epoch):
            start_time = time.time()

            for index, (original_text, sum_text) in enumerate(train_loader):

                original_token = tokenizer.batch_encode_plus(original_text, padding=True, return_tensors='pt').input_ids.to(device)
                sum_token = tokenizer.batch_encode_plus(sum_text, padding=True, return_tensors='pt').input_ids.to(device)

                original_attention_mask = generate_global_attention_mask(tokenizer, original_token).to(device)


                output = model(input_ids = original_token,
                               global_attention_mask = original_attention_mask,
                               labels = sum_token)

                loss = output[0]

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                end_time = time.time()
                train_loss_list.append(loss.item())

                # RESULT PRINT OUT
                if (index+1)%200 == 0:
                    train_loss_mean = sum(train_loss_list) / len(train_loss_list)
                    valid_loss_mean = self.validation(model, self.valid_loader, stop = 30)

                    print("========================================================================================================================")
                    print(f"Batch {(index+1)}  ({((index+1)/len(train_loader))*100 :.3f} %) \t \
                            Train Loss : {train_loss_mean :.4f} \t \
                            Valid Loss : {valid_loss_mean :.4f} \t \
                            Elapsed Time: {(end_time - start_time) :.2f} sec")

                    train_loss_list = []
                    # record_train_loss.append(train_loss_mean)
                    # record_valid_loss.append(valid_loss_mean)

                if (index+1)%1000 == 0:
                    self.save_model_info(model, f"ver_{(index+1)//1000}")

        return model

    ''' ========================================== V A L I D A T I O N ====================================================='''
    def validation(self, model, valid_loader, stop):
      tokenizer = self.tokenizer
      generate_global_attention_mask = self.generate_global_attention_mask

      valid_loss_list = []
      stop = stop
      device = self.device

      model.eval()

      with torch.no_grad():
        for valid_index, (valid_original_text, valid_sum_text) in enumerate(valid_loader):

            original_token = tokenizer.batch_encode_plus(valid_original_text, padding=True, return_tensors='pt').input_ids.to(device)
            sum_token = tokenizer.batch_encode_plus(valid_sum_text, padding=True, return_tensors='pt').input_ids.to(device)

            original_attention_mask = generate_global_attention_mask(tokenizer, original_token).to(device)

            valid_output = model(input_ids = original_token,
                                global_attention_mask = original_attention_mask,
                                labels = sum_token)

            valid_loss = valid_output[0]
            valid_loss_list.append(valid_loss.item())

            if (valid_index+1)%stop == 0:
                break;

      valid_loss_mean = sum(valid_loss_list) / len(valid_loss_list)
      model.train()

      return valid_loss_mean


    ''' ========================================== L O A D  M O D E L ====================================================='''
    def load_model_info(self, version):

      file_path = f"./policy_model/policy_model_{version}.pth"

      if not os.path.exists(file_path):
          print("FATAL ERROR : model path not exist")

      model_info = torch.load(file_path)
      print(f"model_loaded : policy_model_{version}")

      model = self.policy_model
      model.load_state_dict(model_info)
      model.eval()

      return model

    ''' ========================================== I N F E R E N C E  ====================================================='''
    def policy_inference(self, text, model):
      device = self.device
      # model = self.policy_model.to(device)
      # model = self.load_model_info(version)
      tokenizer = self.tokenizer

      text_token = tokenizer.batch_encode_plus(text, padding=True, return_tensors='pt').input_ids.to(device)
      # text_global_att = self.generate_global_attention_mask(tokenizer, text_token).to(device)

      summary_token = model.generate(inputs= text_token)
      summary_token = summary_token

      summarized_text = tokenizer.batch_decode(summary_token, skip_special_tokens=True)

      return summarized_text

"""### Initialize
----
"""

policy = Policy(_batch_size = BATCH_SIZE,
                _epoch= EPOCH,
                _lr = LEARNING_RATE,
                _model_name = "allenai/led-large-16384-arxiv",
                _device = device)

"""### Training
-----
"""

# model = policy.train()

"""### Inference
----
"""

text= "Ok so a bit of back story, my fiancee have been together 6 years. We have one 3 year old daughter together. We have had serious problems the last year. I found out she cheated on me with a coworker (March 2015). I've never been unfaithful to her, but I'm not perfect by any means. I don't believe I was being a good partner to her.. Not that it's any excuse to cheat. \n\nThe problem is we never went to counseling or anything, never really talked about it other than maybe that first week after I found out about it. She has a lot of depression and anxiety issues. We Co parent great, our sex life is good, we don't argue really. She just shuts down sometimes and gives up so to speak.\n\n Two days ago she tells me she just can't do it anymore. She feels hopeless etc. She is a stay at home mom now and money is tight for us with one income which has also caused issues. She says she loves me with all her heart but isn't in love like she was. \n\nAnd I know this all sounds like she's cheating again but I honestly don't think so. Should I let her leave, try to get to counseling? Just don't know what to do. Sorry for the long rambling post."

text = (text,) # 모델안에서 tensor로 바꿔 처리하기 위해 Text를 tuple이나 리스트에 담아야 해요 ㅠㅠ

# policy_model = policy.load_model_info("ver_2")
policy_model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv").to(device)

summary = policy.policy_inference(text, model = policy_model)

"""### Load Model
----
"""

policy_model = policy.load_model_info("ver_1").to(device)

"""### Save Model
----
"""

policy.save_model_info(policy_model, "ver_x")