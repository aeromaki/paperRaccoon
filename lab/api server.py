from fastapi import FastAPI
from typing import Optional, List
import requests
from pydantic import BaseModel
import json

app = FastAPI()

class LanguageSection(BaseModel):
    section: str
    text: str

class LanguageInput(BaseModel):
    text: List[LanguageSection]


@ app.get("/inference")
async def get_inference(input_data: LanguageInput):
    # Send request to the model server
    model_server_endpoint0 = "https://y0hg21gulrpphe-8089.proxy.runpod.net/summarize" # model 나쁜거
    model_server_endpoint1 = "https://y0hg21gulrpphe-8089.proxy.runpod.net/summarize_better" # policy model
    inference_results0 = []
    inference_results1 = []
    async def listappend(lis, summary):
        lis.append(summary)
        return lis
    
    for i in range(len(input_data.text)):
        a = {"text": str(input_data.text[i].text)}
        response0 = requests.post(model_server_endpoint0, json=a)
        response1 = requests.post(model_server_endpoint1, json=a)
    
        if response0.status_code == 200:
            # Process the inference results
            summary0 = response0.json()
            await listappend(inference_results0, summary0["summary"])
            print(inference_results0)

        if response1.status_code == 200:
            # Process the inference results  
            summary1 = response1.json()
            await listappend(inference_results1, summary1["summary"])
            
        result_dict = {
        "inference_results0": inference_results0,
        "inference_results1": inference_results1
    }    
    return result_dict


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
    