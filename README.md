<img src='./raccoon.jpg'>

## Summary

- **현재 데이터**
  - Open AI - summarize_from_feedback [[Link](https://huggingface.co/datasets/openai/summarize_from_feedback)]
  - ArXiv Summarization [[Link](https://huggingface.co/datasets/ccdv/arxiv-summarization)]
- **데이터 형식** 
  - 배포 전 Fine tuning
    - Policy model Training
      - original text, summarized text
    - Reward model Training
      - original text, summarized text 2개, human choice 
        - 정리하면 original text, good summarized text, bad summarized text 필요 
    - RLHF Training
      - original text
  - 서비스 배포 
    - Reward model Training 
      -  original text, summarized text (policy model 2개의 출력 결과), human choice
    - RLHF Training
      - Reward model Training DB에서 original text만 필요 
- **코드 설명** 
  - lab/policy_model_with_training/**Policy_Model_Trainingver2.ipynb**
    - 데이터 로드 및 전처리, Policy 모델 정의 및 Trainining 코드
      - policy model == language model 
  - lab/reward_model/**reward_train.ipynb**
    - 데이터 로드 및 전처리, Reward 모델 정의 및 Training 코드
  - lab/_garb 안에 있는 것들은 신경 쓸 필요 없음 (그냥 저장용, 앞으로 쓸 일 없음)
- **현재 결과**
  - Reward model 구현 완료, Training에서 수렴성 보임 
  - Policy model 구현 완료, Training에서 수렴성 보임
  - RL Training 구현 진행 중 
    - Reward model, Policy 모델은 위의 모델 사용하면 됨 
    - Training만 구현하면 되는 데 시험기간 이후 진행 예정 



## 모델팀 진행계획

- 시험기간 이후 ~ 컨퍼 전까지 
  - RL Training, PDF 전처리, Data Augmentation 코드 완성하고 성능개선 부분에 집중하여 진행 
  - 디버깅 
  - 코드 배포용이하도록 정리 



## 모델팀이 서버팀에게 궁금한 것

1. ipynb 코드 파일 python으로 변환 어떤 형식으로 원하는 지?

   - 어떤 코드를 어떻게 실행하고 싶은 지 궁금

     > ex) python reward_model.py  -> reward model 출력 하도록?
     >
     > ex) python reward_model_train.py -> reward model Training 실행? 

2. 더 보완하거나 추가해야할 부분? 



----

## 데이터

- Open AI - summarize_from_feedback [[Link](https://huggingface.co/datasets/openai/summarize_from_feedback)]
  - Policy model
    - axis의 validation, test 데이터 셋 이용
  - Reward model 
    - comparisions의 train, validation 데이터 셋 이용

- ArXiv Summarization [[Link](https://huggingface.co/datasets/ccdv/arxiv-summarization)]
  - 학습하는데 GPU 약 80G 필요
  - Policy model
    - document의 train, validation, test 데이터 셋 이용 
    - 데이터가 커서 GPU 부족으로 안돌아감. GPU 확보되면 돌려보겠음
  - Reward model
    - document의 train, validation, test 데이터 셋 이용 



## 데이터 형식

- **Policy model** 

  - 아래와 같은 형식의 데이터만 준비되면 학습 가능  

  | original_text (string)                                  | sum_text (string)      |
  | ------------------------------------------------------- | ---------------------- |
  | Good Morining! Today is sunny. I want to play ~         | The author feels good. |
  | Good Afternoon! Today is rainy. I don't want to play ~  | The author feels bad.  |
  | Good Evening! Today is snowing. I want to drive fast. ~ | The author is crazy    |

  - Column 이름은 'orignal_text' (원본텍스트, 영어, string), 'sum_text' (요약텍스트, 영어. string)

- **Reward model**

  - 아래와 같은  형식의 데이터만 준비되면 학습 가능 

    | article (string)                                        | abstract  (dictionary)                                       | choice |
    | ------------------------------------------------------- | ------------------------------------------------------------ | ------ |
    | Good Morining! Today is sunny. I want to play ~         | {text: The author feels good.}<br />{text: The weather is bad. } | 0      |
    | Good Afternoon! Today is rainy. I don't want to play ~  | {text: The author feels bad.} <br />{text: The author is pokemon.} | 0      |
    | Good Evening! Today is snowing. I want to drive fast. ~ | {text: The author want to sing.}<br />{text: The author want to drive.} | 1      |

  - 현재 코드에 구현된 column이름은 'article (원본 텍스트, string)', 'abstract' (좋은 요약본, 안좋은 요약본, dictionary), 'choice' (0 또는 1의 int)으로 이뤄져 있음

  - choice는 Human이 선택한 '좋은 요약 텍스트'를  의미함. 0이면 abstract의 0번째 Text가 좋은 텍스트이고 1번째 Text가 안좋은 텍스트로 분류 되는 것. 

  - 쉽게 이해하면 아래와 같음. 이런 식으로 데이터 형식 만들어도 코드 수정 조금만해서 돌아갈 수 있음. 서버팀 편하신대루. choice 없애고 good_summarize, bad_summarize 이런식으로 나눠도 괜찮.

    | original_text (string)                                  | summarize_0  (string)    | summarize_1 (string)      | choice |
    | ------------------------------------------------------- | ------------------------ | ------------------------- | ------ |
    | Good Morining! Today is sunny. I want to play ~         | The author feels good.   | The weather is bad.       | 0      |
    | Good Afternoon! Today is rainy. I don't want to play ~  | The author feels bad.    | The author want to drive. | 0      |
    | Good Evening! Today is snowing. I want to drive fast. ~ | The author want to sing. | The author want to drive  | 1      |

- **RLHF Training**

  - Policy model과 Reward model 동시에 RLHF로 업데이트 할 때 사용 

    | original_text (string)                                  |
    | ------------------------------------------------------- |
    | Good Morining! Today is sunny. I want to play ~         |
    | Good Afternoon! Today is rainy. I don't want to play ~  |
    | Good Evening! Today is snowing. I want to drive fast. ~ |

  - 'original_text' (원본텍스트, string) 이것만 준비되면 됨

- **정리하면 Policy model, Reward model Fine tuning할 때는 original text, summarized text, human choice (reward model만)이 필요하고 RLHF Training을 진행할 때는 original text만 필요**

  - 실제 서비스할 때는 Reward model online learning과 RLHF Training을 진행
    - **즉, 서비스 할 때는 original text, summarized text (policy model 2개의 출력 결과), human choice로 이뤄진 DB를 지속적으로 업데이트해야함. RLHF Training은 이 DB에서 original_text만 빼오면 됨** 



## 코드 구성

- 코드 설명은 ipynb 파일 내에 주석으로 모두 처리해 놓았습니다. 부족한 부분 말씀해주시면 됩니다.
- **Policy_Model_Training_ver2.ipynb**
  - Import
  - Hyper Parameter setting
  - Tokenizer
  - DataLoad
    - Data preprocessing
    - DataLoader
  - Policy model
    - LEDForConditionalGeneration
  - Training
    - DataLoader에서 데이터를 batch단위로 불러와 학습 진행 
    - batch 1000개 학습할 때마다 모델을 ./policy_model 폴더에 저장
    - Training이 끝나면 model (obejct), record_train_loss (list), record_valid_loss (list) 반환
- **reward_train.ipynb**
  - Import
  - DataLoad
  - Tokenizer
  - Reward model
  - Data Augmentation
  - Training
    - 데이터 배치로 쪼개기, 토큰화 및 augmentation 진행 
    - 모델 학습
    - 배치 1000개마다 현재폴더에 모델 저장