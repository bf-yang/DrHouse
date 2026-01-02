# GPT auto evaluation on the diagnostic dialogues.
# Poe API
import asyncio
import fastapi_poe as fp
from langchain.prompts import ChatPromptTemplate
import os

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
    print(prompt)
    # prompt=[fp.ProtocolMessage(role="user", content="hello world")]
    response_text=asyncio.run(get_responses(api_key, prompt,model_id))
    return response_text

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
model_version = 'GPT-4-Turbo'


# Settings
user_name = 'hgw_t1'
method_list = ['gpt3', 'gpt4', 'llama3', 'huatuo', 'meddm', 'drhouse']

for method in method_list:
    # Load dialogues
    path_dialogues = os.path.join('exp_real/results/dialogues',user_name,'res_'+user_name+'_'+method+'.txt')
    with open(path_dialogues, 'r') as file:
        dialogues = file.read()
    print(dialogues)

    disease = 'Acute bronchitis' # evaluation disesae label
    # Load evaluation guidelines tree
    with open('data/guideline_trees/txt_guidelines/acute_bronchitis.txt', 'r') as file:
        guideline_tree = file.read()
    print(guideline_tree)

    EVAL_TEMPLATE = """
        You are a professional, impartial, and strict scorer, Below is a conversation between a patient and an Al doctor. The patient's ground truth disease is {disease}. Below also has a medical diagnosis guideline for {disease} disease. Based on the medical diagnosis guideline below, rate the doctor's performance according to:
        1. How closely does the AI doctor follow established diagnostic standards in its decision-making process? 
        2. How much does sensor data assist the virtual doctor in making decisions? Please note that the focus is not on the number of sensors involved in the LLM question, but rather on the information derived from the sensor data throughout the entire diagnostic process.
        3. Is the doctor's diagnosis consistent with the patient's ground truth disease?
        Please rate the doctor's performance on a scale of 1-100 and provide an explanation.

        Ground truth disease:
        {disease}

        Guideline tree:
        {guideline_tree}
        
        [start of conversation]
        {conversation}
        [end of conversation]
        """
    # Fill dialogues, disease, and guideline tree into eval_prompt
    prompt_eval_template = ChatPromptTemplate.from_template(EVAL_TEMPLATE)
    prompt_eval = prompt_eval_template.format(disease=disease, 
                                        guideline_tree=guideline_tree,
                                        conversation=dialogues)

    # GPT Evaluation
    input = [{'role':'user','content':prompt_eval}]
    response_text=send_request_poe_api(input,model_id=model_version,api_key=api_key)
    print(f'GPT Evaluation: {response_text}')

    # Save results
    path_res_save = os.path.join('exp_real/results/scores',
                                user_name,method+'.txt')
    with open(path_res_save, 'w') as f:
        for line in response_text:
            f.write(f"{line}")