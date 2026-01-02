# Evaluation the performance of retreive guidelines
# MedDM, use embedding database of raw guideline pdf for retrieval
import pandas as pd
import argparse
from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
openai_api_key = "2c8668618f3f41a5845cdfd3d72ff8b3"

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

    df = df.drop(df[df['label']=='Fungal infection'].index)

    df = df.drop(df[df['label']=='Migraine'].index)

    df = df.drop(df[df['label']=='Malaria'].index)
    df = df.drop(df[df['label']=='Psoriasis'].index)
    df = df.drop(df[df['label']=='Dengue'].index)
    df = df.drop(df[df['label']=='Typhoid'].index)
    df = df.drop(df[df['label']=='allergy'].index)
    df = df.drop(df[df['label']=='diabetes'].index)
    df = df.drop(df[df['label']=='peptic ulcer disease'].index)
    df = df.drop(df[df['label']=='Pneumonia'].index)
    # df = df.drop(df[df['label']=='Hypertension'].index)

    # print(len(df))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--mode", type=str, default='txt', help='txt or pdf')

    args = parser.parse_args()

    # Load symptom dataset
    path = 'src/guideline_tree_retrieval/data/Symptom2Disease.csv'
    df = pd.read_csv(path)
    print(len(df))
    df = remove_disease(df)
    print(len(df))

    # Split train and test
    df = df.sample(frac=1.0,random_state=1).reset_index(drop=True)
    R = 0.7
    N = int(R*len(df))
    df_train = df.loc[0:N-1]
    df_test = df.loc[N:]

    # query guideline
    CHROMA_PATH = "src/guideline_tree_retrieval/data/database"
    embedding_function = AzureOpenAIEmbeddings( 
                openai_api_key=openai_api_key,
                azure_endpoint="https://cuhk-aiot-gpt4.openai.azure.com/",
                azure_deployment = "text-embedding-ada-002"
                )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    cnt = 0
    for idx in range(len(df_test)):
        query_text = df_test.iloc[idx]['text']
        label_gt = df_test.iloc[idx]['label']
        # Search the DB 
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        # print("Sources:", sources)
        label_pred_filepath = sources[0]
        # root = 'src/guideline_tree_retrieval/data/Symptom2Disease_txt/'  # length=54
        label_pred = label_pred_filepath[54:-4]
        print('GT: ', label_gt, '  Pred: ', label_pred)

        if label_gt == label_pred:
            cnt += 1
    print("Accuracy: ", cnt/len(df_test))