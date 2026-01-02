# Virtual doctor cli
# Baseline GPT3.5 or GPT-4
# No retrieval of the medical knowldge (Up-to-date)
import os
import sys
sys.path.append('/home/bufang/DrHouse')
from openai import AzureOpenAI
from langchain.prompts import ChatPromptTemplate
from src.utils import get_dialogue_demos,get_system_prompt_bs
from pprint import pprint
import argparse
api_key = "2c8668618f3f41a5845cdfd3d72ff8b3"

client = AzureOpenAI(
  api_key = api_key,  
  api_version = "2023-05-15",
  azure_endpoint = "https://cuhk-aiot-gpt4.openai.azure.com/"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, default='gpt-35-turbo',
                        help='gpt-35-turbo or gpt-4-1106')
    parser.add_argument("--exp",type=str,default='real', help='real or simulation')
    parser.add_argument("--user_name",type=str,default='cj_t2', help='cj_t2 (real) or sim_data_ab_1 (simulation)')

    args = parser.parse_args()

    if args.exp == 'real':
        path_sensor_db_base = 'exp_real/data/vector_databases'
        user_name = args.user_name  #'cj_t2'
    if args.exp == 'simulation':
        path_sensor_db_base = 'exp_simulation/data/vector_databases'
        user_name = args.user_name  #'sim_data_ab_1'

    # User self-report symptom 
    user_symptom = input("[ Patient description ]: ") 

    # Retrieve top-k similarity dialogue demos
    PATH_DEMO = 'data/medical_dialogues/MedDialog/english-train.json'
    demo_topk = get_dialogue_demos(PATH_DEMO, user_symptom)
    # pprint(demo_topk)

    # System prompt
    system_prompt = get_system_prompt_bs()
    print(system_prompt)
    print('--------------------------------------------------------------------------------------------')


    conversation=[{"role": "system", "content": system_prompt}]
    # pprint(conversation)
    TEMPLATE_PATIENT_REPLY = """
    # Patient:
    {patient_response}
    """
    
    sensor_knowledge = None
    dialogue_results = []
    cnt_all = 0
    MAX_ROUND = 10
    user_input = user_symptom
    while True:
        if cnt_all == MAX_ROUND:
            break
        if cnt_all > 0:
            user_input = input("[ Patient Symptoms ]: \n") 
        if  user_input == 'end':
            break

        # Combine patient reply and knowledge
        patient_reply_template = ChatPromptTemplate.from_template(TEMPLATE_PATIENT_REPLY)
        user_input_w_knowledge = patient_reply_template.format(patient_response=user_input)
        # pprint(user_input_w_knowledge)

        # Virtual doctor response
        conversation.append({"role": "user", "content": user_input_w_knowledge})
        response = client.chat.completions.create(
            # model="gpt-35-turbo", # model = "deployment_name".
            # model="gpt-4-1106",
            model=args.model_version,
            messages=conversation
        )
        conversation.append({"role": "assistant", "content": response.choices[0].message.content})
        print("[ Virtual Doctor ]: \n" + response.choices[0].message.content + "\n")

        # Save dialogue
        dialogue_results.append("[ Patient ]: " + user_input)
        dialogue_results.append("[ Doctor ]: " + response.choices[0].message.content)
        cnt_all += 1

    print("End successful!")

    # Save dialogues
    if args.model_version == "gpt-4-1106":
        method = 'gpt4'
    if args.model_version == "gpt-35-turbo":
        method = 'gpt3'

    if args.exp == 'real':
        path_res_save = os.path.join('exp_real/results/dialogues',
                                    user_name,'res_'+user_name+'_'+method+'.txt')
    if args.exp == 'simulation':
        path_res_save = os.path.join('exp_simulation/results/dialogues',
                            user_name,'res_'+user_name+'_'+method+'.txt')
        
    with open(path_res_save, 'w') as f:
        for line in dialogue_results:
            f.write(f"{line}\n")

