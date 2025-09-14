# import transformers
# import torch
import requests
import json
import uuid
import traceback
from deepeval.models import DeepEvalBaseLLM


"""使用自定义 LLM 进行评估
https://docs.confident-ai.com/guides/guides-using-custom-llms#json-confinement-for-custom-llms
"""


class Li_Custom_LLM(DeepEvalBaseLLM):
    def __init__(self):
        self.bcs_apihub_request_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "..."))
        self.x_chj_gwtoken = "..."
        self.url = "..."
        self.headers = {
            "BCS-APIHub-RequestId": self.bcs_apihub_request_id,
            "X-CHJ-GWToken": self.x_chj_gwtoken,
            "Content-Type": "application/json",
        }

    def load_model(self):
        # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        return None

    def inference(self, prompt, **kwargs):
        payload = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ], 
                **kwargs
            }
        )
        response = requests.request(
            method="POST", 
            url=self.url, 
            headers=self.headers, 
            data=payload  # Pass any additional arguments to the OpenAI API
        )
        json_data = response.json()
        return json_data["choices"][0]["message"]["content"]
        # return json_data

    def generate(self, prompt: str, **kwargs) -> str:
        return self.inference(prompt, **kwargs)

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def get_model_name(self):
        return "LiAI"


if __name__ == "__main__":

    prompt = "Who is the current president of the United States?"

    model = Li_Custom_LLM()
    print(model.generate(prompt))

    # # Finally, supply it to a metric to run evaluations using your custom LLM:
    # from deepeval.metrics import AnswerRelevancyMetric
    # metric = AnswerRelevancyMetric(model=Li_Custom_LLM)
    # metric.measure(...)

