# Virtual doctor cli
# Baseline method MedDM.
import os
import sys
sys.path.append('/home/bufang/DrHouse')
from openai import AzureOpenAI
from langchain.prompts import ChatPromptTemplate
from src.utils import get_dialogue_demos, get_medical_knowledge,\
    get_system_prompt_meddm, get_guideline_trees_meddm
from pprint import pprint
import argparse
import torch
import os
api_key = "2c8668618f3f41a5845cdfd3d72ff8b3"

client = AzureOpenAI(
  api_key = api_key,  
  api_version = "2023-05-15",
  azure_endpoint = "https://cuhk-aiot-gpt4.openai.azure.com/"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, default='gpt-4-1106',
                        help='gpt-35-turbo or gpt-4-1106')
    parser.add_argument("--exp",type=str,default='real', help='real or simulation')
    parser.add_argument("--user_name",type=str,default='cj_t2', help='cj_t2 (real) or sim_data_ab_1 (simulation)')
    args = parser.parse_args()

    # Path of the patient's sensor data embedding database.
    if args.exp == 'real':
        path_sensor_db_base = 'exp_real/data/vector_databases'
        user_name = args.user_name  #'cj_t2'
    if args.exp == 'simulation':
        path_sensor_db_base = 'exp_simulation/data/vector_databases'
        user_name = args.user_name  #'sim_data_ab_1'
    path_sensor_db = os.path.join(path_sensor_db_base,user_name)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # User self-report symptom 
    user_symptom = input("[ Patient description ]: ") 

    # Retrieve top-k similarity dialogue demos
    PATH_DEMO = 'data/medical_dialogues/MedDialog/english-train.json'
    demo_topk = get_dialogue_demos(PATH_DEMO, user_symptom)
    # pprint(demo_topk)

    # Retrieve top-1 similarity guideline tree
    path_guideline = 'data/guideline_trees/txt_guidelines' # path of guideline tree txt files (not used)
    path_guideline_vdb = 'data/guideline_trees/vector_databases'  # path of vector DB (used)
    guideline_trees = get_guideline_trees_meddm(path_guideline_vdb, user_symptom)

    # System prompt
    prompt_path = "src/baselines/prompt/meddm"
    system_prompt = get_system_prompt_meddm(guideline_trees, prompt_path)
    print(system_prompt)
    print('--------------------------------------------------------------------------------------------')
    # #### Demo Phase #####


    conversation=[{"role": "system", "content": system_prompt}]
    # pprint(conversation)
    TEMPLATE_PATIENT_REPLY = """
    # Patient:
    {patient_response}
        """
    
    sensor_knowledge = None
    dialogue_results = []
    cnt_sensor_retrieval = 0
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
        # Retrieve up-to-date knowledge
        medical_knowledge = get_medical_knowledge(user_input, sensor_knowledge)
        
        # Combine patient reply and knowledge
        patient_reply_template = ChatPromptTemplate.from_template(TEMPLATE_PATIENT_REPLY)
        user_input_w_knowledge = patient_reply_template.format(patient_response=user_input, 
                                            knowledge_med=medical_knowledge,
                                            knowledge_sensor=sensor_knowledge)
        # pprint(user_input_w_knowledge)

        # Virtual doctor response
        conversation.append({"role": "user", "content": user_input_w_knowledge})
        response = client.chat.completions.create(
            # model="gpt-35-turbo", # model = "deployment_name".
            # model="gpt-4-1106",
            model=args.model_version,
            messages=conversation,
            temperature=0.5
        )
        conversation.append({"role": "assistant", "content": response.choices[0].message.content})
        print("[ Virtual Doctor ]: \n" + response.choices[0].message.content + "\n")

        # Save dialogue
        if not sensor_knowledge:
            dialogue_results.append("[ Patient ]: " + user_input)
        else:
            dialogue_results.append("[ Patient ]: " + user_input+ 
                                    " (retrieved knowledge from sensor database: "+sensor_knowledge+")")
        dialogue_results.append("[ Doctor ]: " + response.choices[0].message.content)
        cnt_all += 1

    print("End successful!")
    method = 'meddm'
    if args.exp == 'real':
        path_res_save = os.path.join('exp_real/results/dialogues',
                                    user_name,'res_'+user_name+'_'+method+'.txt')
    if args.exp == 'simulation':
        path_res_save = os.path.join('exp_simulation/results/dialogues',
                            user_name,'res_'+user_name+'_'+method+'.txt')

    with open(path_res_save, 'w') as f:
        for line in dialogue_results:
            f.write(f"{line}\n")

