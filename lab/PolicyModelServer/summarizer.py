from transformers import AutoModel, AutoTokenizer, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, LEDTokenizer
import torch

class Policy():
    def __init__(self,
                 _batch_size,
                 _epoch,
                 _lr,
                 _device,
                 _model_version,
                 # _model_name = "allenai/led-large-16384-arxiv",
                 ):

        # Device
        self.device = _device

        # Tokenizer
        self.tokenizer = self.policy_tokenizer(model_name = _model_version)

        # huggingface
        self.policy_model = self.policy_model(model_name = _model_version)

        # # Model update
        # self.policy_model = self.load_model_info(version = _model_version)

        print(f"\n===================== POLICY INIT COMPLETE =====================\n")

    ''' ========================================= L O A D  M O D E L ===================================================='''
    def policy_model(self, model_name):
      policy_model = LEDForConditionalGeneration.from_pretrained(model_name).to(self.device)

      return policy_model

    ''' ========================================= T O K E N I Z E R ===================================================='''

    def policy_tokenizer(self, model_name):
      tokenizer = LEDTokenizer.from_pretrained(model_name)
      return tokenizer

    # ''' ========================================== G E T  M O D E L ====================================================='''
    # def load_model_info(self, version):

    #   file_path = f"policy_model_{version}.pth"

    #   # if not os.path.exists(file_path):
    #   #     print("FATAL ERROR : model path not exist")

    #   model_info = torch.load(file_path)
    #   print(f"model_loaded : policy_model_{version}")

    #   model = self.policy_model
    #   model.load_state_dict(model_info['model_state_dict'])
    #   model.eval()

    #   return model

    ''' ========================================== I N F E R E N C E  ====================================================='''
    def policy_inference(self, text):
      device = self.device

      tokenizer = self.tokenizer
      model = self.policy_model
      # input_id_2 = tokenizer_2(ARTICLE, return_tensors="pt").input_ids
      # text_token = tokenizer.batch_encode_plus(text, padding=True, return_tensors='pt').input_ids.to(device)
      text_token = tokenizer.batch_encode_plus(text, return_tensors='pt').input_ids.to(device)
      # text_global_att = self.generate_global_attention_mask(tokenizer, text_token).to(device)

      summary_token = model.generate(text_token)

      summarized_text = tokenizer.batch_decode(summary_token, skip_special_tokens=True)

      return summarized_text