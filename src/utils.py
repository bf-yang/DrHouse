import json
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from bert_score import BERTScorer
import torch
import heapq
import pandas as pd
import os
openai_api_key="4d2ff10a8c3d4d09883a4411832b6718"

embedding_function = AzureOpenAIEmbeddings( 
        openai_api_key=openai_api_key,
        azure_endpoint="https://cuhk-aiot-gpt4.openai.azure.com/",
        azure_deployment = "text-embedding-ada-002"
        )


def read_txt(file):
    with open(file,"r") as f:
        data = f.read()
        # print(data)
    return data

def get_dialogue_demos(PATH_DEMO, query):
    # Find the top-k similar dialogue for the query
    # Read dialogue database
    f = open(PATH_DEMO)
    demo_all = json.load(f)
    f.close()

    # Define embedding model
    torch.cuda.set_device(1)
    scorer = BERTScorer(model_type='bert-base-uncased')

    score_list = []
    for idx in range(len(demo_all)):
        demo = demo_all[idx]['description']
        P, R, F1 = scorer.score([query], [demo])
        score_list.append(F1.item())
        # print(F1.item())

    topk_idx = heapq.nlargest(3, range(len(score_list)), score_list.__getitem__)
    # print("Top-K dialogue index: ", topk_idx)

    demo_topk = []
    for idx in topk_idx:
        demo_topk.append(demo_all[idx])
    return demo_topk



INSTRUCTIONS_TEMPLATE = """
    # OVERALL INSTRUCTIONS
    {instruction_overall}

    # TASK INSTRUCTIONS
    {instruction_task}

    # GUIDELINE FOR DIAGNOSIS
    {guideline}

    Please refer to the following demos for multi-round diagnosis:
    # INPUT DATA
    {input_data}
    """
def get_system_prompt(guideline_trees,disease_list,simscore_list,prompt_path):
    # Add the retrieved potential diseases and similarity scores in the system prompt
    TASK_INSTRUCTIONS_TEMPLATE = read_txt(os.path.join(prompt_path,"instruction_task.txt"))
    task_instruction_template = ChatPromptTemplate.from_template(TASK_INSTRUCTIONS_TEMPLATE)
    instruction_task = task_instruction_template.format(disease_1=disease_list[0],
                                        disease_2=disease_list[1],
                                        disease_3=disease_list[2],
                                        disease_1_prior = simscore_list[0],
                                        disease_2_prior = simscore_list[1],
                                        disease_3_prior = simscore_list[2],
                                        N=len(disease_list)
                                        )
    # Add overall instruction
    instruction_overall = read_txt(os.path.join(prompt_path,"instruction_overall.txt"))
    # Add medical dialogue demos (now fixed)
    input_data = read_txt(os.path.join(prompt_path,"fix_demos.txt"))
    instruction_template = ChatPromptTemplate.from_template(INSTRUCTIONS_TEMPLATE)
    system_prompt = instruction_template.format(instruction_overall=instruction_overall,
                                                instruction_task=instruction_task,
                                                guideline =  guideline_trees,
                                                input_data=input_data)
    # print(system_prompt[8:])  # delete first string "Human:/n"
    return system_prompt[8:]


RETRIEVE_MEDICAL_TEMPLATE = """
# PATIENT SYMPTOMS
{patient_symptoms}

# SENSOR DATA KNOWLEDGE
{sensor_knowledge}
    """
def get_medical_knowledge(query_text, sensor_knowledge):
    # first merge the user input with the sensor data knowledge
    # use the merged on to retriev medical knowledge
    retrieve_medical_template = ChatPromptTemplate.from_template(RETRIEVE_MEDICAL_TEMPLATE)
    retrieve_medical_prompt = retrieve_medical_template.format(patient_symptoms=query_text,
                                        sensor_knowledge=sensor_knowledge,)
    # print(retrieve_medical_prompt)
    CHROMA_PATH = "data/medical_database_chroma"
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the medical DB 
    results = db.similarity_search_with_relevance_scores(retrieve_medical_prompt, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results when retrieving medical knowledge.")
        return
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text


SUMMARY_SENSOR_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""
def get_sensordata_knowledge(query_text, path_db, summarize):
    # Sensor data retrieval
    CHROMA_PATH = path_db
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # Search the sensor DB
    results = db.similarity_search_with_relevance_scores(query_text, k=1)
    if len(results) == 0 or results[0][1] < 0.1:
        print(f"Unable to find matching results when retrieving sensor knowledge.")
        return None
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    if not summarize:
        return context_text  # return raw structured data   
    else:
        # use another LLM to summarize the retrieved sensor data from tabluar data to a sentence
        prompt_template = ChatPromptTemplate.from_template(SUMMARY_SENSOR_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(prompt)
        model = AzureChatOpenAI(
            openai_api_key="4d2ff10a8c3d4d09883a4411832b6718",
            azure_endpoint="https://cuhk-aiot-gpt4.openai.azure.com/",
            openai_api_version="2023-05-15",
            azure_deployment="gpt-35-turbo",  
            )
        response_text = model.predict(prompt)
        print('response_text:  ', response_text)
        return response_text # return summarized sentence
    
   
    

def get_guideline_trees(path_vdb, path_guideline_tree, query_text, mode):
    # Guideline trees retrieval
    # PATH_WDB: path of vector DB
    # path_guideline_tree: path of guideline tree files
    if mode == 'MedDM':
        db = Chroma(persist_directory=path_vdb, embedding_function=embedding_function)
        # Search the DB 
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.3:
            print(f"Unable to find matching results.")
            return
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        print("Sources:", sources)

        path_guidelines_tree = sources[0] # path of the retrieved guideline tree
        with open(path_guidelines_tree,'r',encoding='utf-8') as f:
            guidelines_tree = f.read()      

    if mode == 'mapping-based':
        path = 'data/Symptom2Disease/real.csv'
        df = pd.read_csv(path)
        df = remove_disease(df)
        dict_symp2dis = {}
        for idx in range(len(df)):
            dict_symp2dis[df.iloc[idx]['text']] = df.iloc[idx]['label']
        # pprint(dict_symp2dis)
        symptom_keys = list(dict_symp2dis.keys())

        # Define embedding model
        torch.cuda.set_device(1)
        scorer = BERTScorer(model_type='bert-base-uncased')

        # Similarity score
        score_list = []
        for idx in range(len(symptom_keys)):
            symptom = symptom_keys[idx]
            P, R, F1 = scorer.score([query_text], [symptom])
            score_list.append(F1.item())

        # topk_idx = heapq.nlargest(K, range(len(score_list)), score_list.__getitem__)
        K,K_ALL = 3,len(score_list)
        topk_idx_all = heapq.nlargest(K_ALL, range(len(score_list)), score_list.__getitem__)
        topk_idx, topk_disease = [], []
        for idx in topk_idx_all:
            if dict_symp2dis[symptom_keys[idx]] not in topk_disease:
                topk_disease.append(dict_symp2dis[symptom_keys[idx]])
                topk_idx.append(idx)

        guidelines_tree = ''
        disease_list = []
        simscore_list = []
        for k in range(K):
            print("Top-%d similarity symptom: " % k, symptom_keys[topk_idx[k]])
            print("Corresponding disease: ", dict_symp2dis[symptom_keys[topk_idx[k]]], 
                  ",  "
                  "Score: ", score_list[topk_idx[k]])
            
            path = os.path.join(path_guideline_tree,dict_symp2dis[symptom_keys[topk_idx[k]]]+'.txt')
            with open(path,'r',encoding='utf-8') as f:
                guidelines_tree_k = f.read()
                guidelines_tree += '----Diagnosis guideline tree for disease: '+dict_symp2dis[symptom_keys[topk_idx[k]]]+'----\n'+guidelines_tree_k+'\n'

            disease_list.append(dict_symp2dis[symptom_keys[topk_idx[k]]])
            simscore_list.append(score_list[topk_idx[k]])
        # disease_list = [dict_symp2dis[symptom_keys[topk_idx[0]]],
        #                 dict_symp2dis[symptom_keys[topk_idx[1]]],
        #                 dict_symp2dis[symptom_keys[topk_idx[2]]]]
        # print("top-k possible diseases: ", disease_list)

    return guidelines_tree,disease_list, simscore_list



def remove_disease(df):
    # we can not find the corresponding guidelines on UpToData for some diseases
    # so this function is to remove these diseases
    df = df.drop(df[df['label']=='Jaundice'].index)
    # print(len(df))

    df = df.drop(df[df['label']=='drug reaction'].index)
    # print(len(df))

    df = df.drop(df[df['label']=='Common Cold'].index)
    # print(len(df))

    df = df.drop(df[df['label']=='Varicose Veins'].index)
    # print(len(df))

    df = df.drop(df[df['label']=='Impetigo'].index)
    # print(len(df))

    df = df.drop(df[df['label']=='Chicken pox'].index)
    # print(len(df))
    return df


def adaptive_retrieval(text,tokenizer,model_ar):
    text_tokenize = tokenizer(text,padding='max_length',max_length=104,truncation=True,return_tensors="pt")
    output = model_ar(text_tokenize['input_ids'],text_tokenize['attention_mask'])
    prob, pred = torch.max(output, dim=1) 
    return pred


# Baseline's system prompt, i.e., GPT-3.5 and GPT-4
def get_system_prompt_bs():
    # No guideline, but have few-shot demos
    INSTRUCTIONS_TEMPLATE = """
    # OVERALL INSTRUCTIONS
    {instruction_overall}

    # TASK INSTRUCTIONS
    {instruction_task}

    Please refer to the following demos for diagnosis:
    # INPUT DATA
    {input_data}
    """

    instruction_overall = read_txt("src/baselines/prompt/baseline/instruction_overall.txt")
    instruction_task = read_txt("src/baselines/prompt/baseline/instruction_task.txt")
    input_data = read_txt("src/baselines/prompt/baseline/fix_demos.txt")
    instruction_template = ChatPromptTemplate.from_template(INSTRUCTIONS_TEMPLATE)
    system_prompt = instruction_template.format(instruction_overall=instruction_overall,
                                        instruction_task=instruction_task,
                                        input_data=input_data)
    # print(system_prompt[8:])  # delete first string "Human:/n"
    return system_prompt[8:]


# Baseline's system prompt for MedDM
# https://arxiv.org/abs/2312.02441
def get_system_prompt_meddm(guideline_trees,prompt_path):
    # Prompt for baseline MedDM.
    instruction_task0 = read_txt(os.path.join(prompt_path,"instruction_task.txt"))

    INSTRUCTIONS_TEMPLATE = """
    # OVERALL INSTRUCTIONS
    {instruction_overall}

    # TASK INSTRUCTIONS
    {instruction_task}

    # GUIDELINE FOR DIAGNOSIS
    {guideline}

    Please refer to the following demos for multi-round diagnosis:
    # INPUT DATA
    {input_data}
    """

    instruction_overall = read_txt(os.path.join(prompt_path,"instruction_overall.txt"))
    instruction_task = instruction_task0
    input_data = read_txt(os.path.join(prompt_path,"fix_demos.txt"))
    instruction_template = ChatPromptTemplate.from_template(INSTRUCTIONS_TEMPLATE)
    system_prompt = instruction_template.format(instruction_overall=instruction_overall,
                                                instruction_task=instruction_task,
                                                guideline =  guideline_trees,
                                                input_data=input_data)
    # print(system_prompt[8:])  # delete first string "Human:/n"
    return system_prompt[8:]


# Baseline approach for guideline tree retrieval (MedDM uses RAG)
def get_guideline_trees_meddm(PATH_VDB, query_text):
    # PATH_VDB: path of vector DB   
    db = Chroma(persist_directory=PATH_VDB, embedding_function=embedding_function)
    # Search the DB 
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.3:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    print("Sources:", sources)
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    disease_list = [sources[0][32:-4],sources[1][32:-4],sources[2][32:-4]]
    print(disease_list)
    path_guidelines_tree = sources[0] # path of the retrieved guideline tree
    with open(path_guidelines_tree,'r',encoding='utf-8') as f:
        guidelines_tree = f.read()

    guidelines_tree = ''
    K = 3
    for k in range(K):
        # path = os.path.join(PATH_FILES,dict_symp2dis[symptom_keys[topk_idx[k]]]+'.txt')
        path = sources[k]
        with open(path,'r',encoding='utf-8') as f:
            guidelines_tree_k = f.read()
            guidelines_tree += '----Diagnosis guideline tree for disease: '+disease_list[k]+'----\n'+guidelines_tree_k+'\n'
    return guidelines_tree