# DrHouse cli
import os
import sys
sys.path.append('/home/bufang/DrHouse')
from openai import AzureOpenAI
from langchain.prompts import ChatPromptTemplate
from utils import get_dialogue_demos,get_system_prompt,get_medical_knowledge,\
    get_guideline_trees,adaptive_retrieval,get_sensordata_knowledge
import argparse
import torch
from transformers import BertTokenizer, BertTokenizer
from semantic_filter import BertClassifier
import os
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()

api_key = "4d2ff10a8c3d4d09883a4411832b6718"

client = AzureOpenAI(
  api_key = api_key,  
  api_version = "2023-05-15",
  azure_endpoint = "https://cuhk-aiot-gpt4.openai.azure.com/"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, default='gpt-4-1106',
                        help='gpt-35-turbo or gpt-4-1106')
    parser.add_argument("--mode_guideline_tree", type=str, default='mapping-based',
                        help='mapping-based: mapping-based approach for retrieval, MedDM: guideline tree vector DB')
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

    # System settings
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Propmt path
    prompt_path = "prompt"
    path_demo = 'data/medical_dialogues/MedDialog/english-train.json'
    path_guideline = 'data/guideline_trees/txt_guidelines' # path of guideline tree txt files
    path_guideline_vdb = 'xxx'  # path of vector DB
    
    # Adaptive retrieval settings
    model_ar = BertClassifier(flag="base")  # ar: adaptive retrieval
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    save_path = 'checkpoints/semantic_filter.pt'
    model_ar.load_state_dict(torch.load(save_path))
    
    # Information fusion template
    TEMPLATE_PATIENT_REPLY = """
    # Patient:
    {patient_response}

    # Knowledge retrieved from medical textbooks
    {knowledge_med}

    # Knowledge retrieved from sensor data
    {knowledge_sensor}
    """
    

    # Start medical consultation
    n_turn = 0         # n-th turn
    cnt_sensor_retrieval = 0 # number of sensor data retrieval
    symptom_total = '' # total symptoms reported by each turn
    sensor_knowledge = None
    dialogues = []     # record dialogues and save
    while True:
        user_input = input("[ Patient Symptoms ]: \n") 
        # Dialogue demos retrieval
        if n_turn == 0:
            demo_topk = get_dialogue_demos(path_demo, user_input)
        if user_input == 'end': # End consultation
            break

        symptom_total += user_input # cumulative symptoms

        # Retrieve medical knowledge (retrieved by user current input)
        medical_knowledge = get_medical_knowledge(user_input, sensor_knowledge)
        # Merge medical knoweldge, n-1 turn sensor data with current user input
        patient_reply_template = ChatPromptTemplate.from_template(TEMPLATE_PATIENT_REPLY)
        user_input_merge = patient_reply_template.format(patient_response=user_input, 
                                            knowledge_med=medical_knowledge,
                                            knowledge_sensor=sensor_knowledge)        

        # Retrieve guideline trees (retrieved by user total symptoms)
        guideline_trees,disease_list, simscore_list = get_guideline_trees(path_guideline_vdb, 
                                    path_guideline, symptom_total, args.mode_guideline_tree)

        # LLM Decision-making
        # Update system prompt
        system_prompt = get_system_prompt(guideline_trees,disease_list, simscore_list, prompt_path)
        if n_turn == 0:
            conversation=[{"role": "system", "content": system_prompt}]  # initailize conversation
        else:
            conversation[0]['content'] = system_prompt # update system prompt (update guideline trees)
        # LLM inference
        conversation.append({"role": "user", "content": user_input_merge})
        response = client.chat.completions.create(
            model=args.model_version,
            messages=conversation,
            temperature=0.5
        )
        conversation.append({"role": "assistant", "content": response.choices[0].message.content})
        print("[ DrHouse ]: \n" + response.choices[0].message.content + "\n")

        # Save dialogue
        if not sensor_knowledge:
            dialogues.append("[ Patient ]: " + user_input)
        else:
            dialogues.append("[ Patient ]: " + user_input+ 
                                    " (retrieved knowledge from sensor database: "+sensor_knowledge+")")
        dialogues.append("[ DrHouse ]: " + response.choices[0].message.content)

        # Retrieve sensor data
        # Adaptive Retrieval
        pred = adaptive_retrieval(response.choices[0].message.content, # doctor text
                                  tokenizer,
                                  model_ar)
        if pred.item():
            print("Start Retrieval")
            sensor_knowledge = get_sensordata_knowledge(response.choices[0].message.content, path_sensor_db, summarize=False)
            cnt_sensor_retrieval += 1
        else:
            print("No Sensor Data Retrieval")

        n_turn += 1

    dialogues.append('\n')
    dialogues.append(["cnt_sensor_retrieval: ", cnt_sensor_retrieval])
    dialogues.append(["n_turn: ", n_turn])

    # Save dialogues
    print("End successful!")
    method = 'drhouse'
    if args.exp == 'real':
        path_res_save = os.path.join('exp_real/results/dialogues',
                                    user_name,'res_'+user_name+'_'+method+'.txt')
    if args.exp == 'simulation':
        path_res_save = os.path.join('exp_simulation/results/dialogues',
                            user_name,'res_'+user_name+'_'+method+'.txt')
    with open(path_res_save, 'w') as f:
        for line in dialogues:
            f.write(f"{line}\n")