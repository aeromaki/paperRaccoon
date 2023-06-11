import pickle
from tqdm import tqdm
from transformers import LEDTokenizer
import pandas as pd

import sys
dataset_name = sys.argv[1]

tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

with open(f"{dataset_name}.pickle", "rb") as f:
        dataset = pickle.load(f)

for i in tqdm(dataset):
        for j in i:
                i[j] = tokenizer.decode(i[j], skip_special_tokens=True)

dataset = pd.DataFrame(dataset)
dataset.to_pickle(f"{dataset_name}_str.pickle")