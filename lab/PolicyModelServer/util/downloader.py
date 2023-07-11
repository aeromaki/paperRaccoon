import gdown

url = 'https://drive.google.com/file/d/13AJNdIcUjsa3EXKIJvJrzALNf4Fr6y1V/view?usp=sharing'
output = 'reward_model_openai_2190.pth'
gdown.download(url, output, quiet=False, fuzzy=True) 