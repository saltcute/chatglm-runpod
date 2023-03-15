import runpod
import os
import json
from transformers import AutoTokenizer, AutoModel

sleep_time = int(os.environ.get('SLEEP_TIME', 3))

# load your model(s) into vram here

tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "./model", trust_remote_code=True).half().cuda()
model.eval()


def handler(event):
    # do the things
    try:
        input = event['input']
    except:
        return 'Generation failure, no input'

    try:
        prompt = input['prompt']
    except:
        return 'Generation failure, no prompt'

    try:
        history = json.load(input['history'])
    except:
        history = None

    try:
        response, history = model.chat(tokenizer, prompt, history=history)
    except:
        return 'Generation failure'

    return {
        response,
        json.dump(history)
    }


runpod.serverless.start({
    "handler": handler
})
