# Evaluation on the performance of LLM on Adaptive Retrieval
import pandas as pd
import logging
import numpy as np
from langchain.prompts import ChatPromptTemplate
from openai import AzureOpenAI
from langchain.prompts import ChatPromptTemplate
import asyncio
import fastapi_poe as fp
import time
# import nest_asyncio # The two lines are only for Jupyter (error of asyncio)
# nest_asyncio.apply()
import random
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default='GPT-3.5-Turbo')
parser.add_argument("--few_shot", type=int, default='4')
args = parser.parse_args()

def prompt_covert_poe(message):
    """
    fp.ProtocolMessage(role="user", content="hello world")
    return list[ProtocolMessage]
    """
    prompt = []
    for item in message:
        role = item["role"]
        if role == "assistant":
            role="bot"  
        content = item["content"]
        prompt.append(fp.ProtocolMessage(role=role, content=content))
    return prompt

# Create an asynchronous function to encapsulate the async for loop
async def get_responses(api_key, messages,model_id):
    # pdb.set_trace()
    response=""""""
    async for partial in fp.get_bot_response(messages=messages, bot_name=model_id, api_key=api_key):
        response=response+partial.text
    return response

def send_request_poe_api(message,  model_id="Claude-3.5-Sonnet", api_key=""):
    # pdb.set_trace()
    prompt=prompt_covert_poe(message)
    # print(prompt)
    # prompt=[fp.ProtocolMessage(role="user", content="hello world")]
    response_text=asyncio.run(get_responses(api_key, prompt,model_id))
    return response_text

def read_examples(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all lines from the file and strip newline characters
        lines = [line.strip() for line in file.readlines()]
    return lines

api_key = 'loQuMMWmxUm9noYg5bNX4gvKnmL8yoewbQVON_nM2hM'
model_ids=[
    "GPT-3.5-Turbo",
    'GPT-4o',
    'GPT-4-Turbo',
    'Claude-3-Opus',
    "Claude-3.5-Sonnet",
    'Claude-3-Sonnet',
    'Mistral-Large'
]

# model_id = 'GPT-3.5-Turbo'
model_id = args.model_id
logging.basicConfig(filename='src/adaptive_retrieval/results/llm_'+model_id+'_fewshotsensor_'+str(args.few_shot)+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

filename_test = 'src/adaptive_retrieval/data/test.xlsx'
df = pd.read_excel(filename_test, sheet_name='dataset')
queries = df['text']
gt = df['label']

random_seed = 1999
setup_seed(random_seed)
# Read examples
file_path = 'src/adaptive_retrieval/data/examples.txt'
examples = read_examples(file_path)
n = args.few_shot
selected_examples = random.sample(examples, n)
selected_examples = '\n'.join(selected_examples)

# Set prompt
PROMPT_TEMPLATE = """
You have been assigned the task of text classification. This task is to determine whether to initiate sensor database retrieval based on the questions of virtual doctors. Sensor database contains heart rate, mood, body temperature, walking distance, sleep score, SpO2, and respiratory rate. Your objective is to classify a given text into one of several possible class labels, based on the text. Your output should consist of a single class label that indicates whether to start retrieval of sensor data or not. Choose ONLY from the given class labels below and ONLY output the label without any other characters.
# Text:
{query}

# Examples:
{example}

# Labels:
Retrieval, Not Retrieval

# Answer:

"""
cls2name = {'Retrieval':1,'Not Retrieval':0}
res = []
corr = 0
for idx in range(len(queries)):
# for idx in range(1):
    time.sleep(60)
    print(30*'*','Number of Query:', idx, 30*'*')
    query = queries[idx]
    print("Q: ", query)

    # Add query into the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(query=query, example=selected_examples)
    # print(prompt)
    # LLM inference
    input = [{'role':'user','content':prompt}]
    response=send_request_poe_api(input,model_id=model_id,api_key=api_key)
    print("A: ",response)
    res.append(response)

    # Accuracy
    pred = response
    GT = df['label'][idx]
    if cls2name[pred] == GT:
        corr += 1
        print('Ground Truth: ', GT, 'Corrected')
    else:
        print('Ground Truth: ', GT, 'Error')

    logging.info(f'SampleID {idx}, Q: {query}, A: {pred}, GT: {GT}')

acc =  corr/len(queries)
print("Accuracy: ", acc)
# Save results 
res.append(["Accuracy: ", acc])
with open("src/adaptive_retrieval/results/llm_"+model_id+"_fewshotsensor_"+str(args.few_shot)+".txt", "w") as file:
    # Write each item on a new line
    for item in res:
        file.write(f"{item}\n")