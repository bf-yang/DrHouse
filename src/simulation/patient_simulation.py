# Simulation of patient.
import sys
sys.path.append('/home/bufang/DrHouse')
from openai import AzureOpenAI
from langchain.prompts import ChatPromptTemplate
import argparse
import torch
import os
import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()

api_key = "4d2ff10a8c3d4d09883a4411832b6718"

# LLM agent patient
client_patient = AzureOpenAI(
  api_key = api_key,  
  api_version = "2023-05-15",
  azure_endpoint = "https://cuhk-aiot-gpt4.openai.azure.com/"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, default='gpt-4-1106',
                        help='gpt-35-turbo or gpt-4-1106')
    args = parser.parse_args()

    # Set patient agent
    symptoms = 'I have a fever, yellow sputum, chilled, sore throat, sore throat, muscle aches, negative COVID-19.'
    # Read fixed prompt
    file_path = 'src/simulation/prompt/patient_simulation/prompt_patient.txt'
    with open(file_path, 'r') as file:
        prompt_patient = file.read()
    # Add symptoms to prompt 
    prompt_patient = ChatPromptTemplate.from_template(prompt_patient)
    prompt_patient = prompt_patient.format(symptoms=symptoms) 
    conversation_patient=[{"role": "system", "content": prompt_patient}]

    # System settings
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Start medical consultation
    MAX_TURN = 10
    n_turn = 0         # n-th turn
    while n_turn < MAX_TURN:
        # First patient report their symptoms.
        patient_output = client_patient.chat.completions.create(
            model=args.model_version,
            messages=conversation_patient,
            temperature=0.5
        )
        patient_output = patient_output.choices[0].message.content
        conversation_patient.append({"role": "system", "content": patient_output})
        print("[ Patient Symptoms ]: \n", patient_output)

        # Next doctor input their questions or summary.
        doctor_input = input("[ Doctor's Input ]: \n")
        conversation_patient.append({"role": "system", "content": doctor_input})
        print(conversation_patient)




   