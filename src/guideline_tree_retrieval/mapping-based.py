# Evaluation the performance of retreive guidelines
# Mapping-based approach (ours, query transform)
import pandas as pd
import argparse
import pandas as pd
from pprint import pprint
from bert_score import BERTScorer
import torch
import heapq

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

if __name__ == "__main__":
    # Load symptom dataset
    path = 'src/guideline_tree_retrieval/data/Symptom2Disease.csv'
    df = pd.read_csv(path)
    print(len(df))
    df = remove_disease(df)
    print(len(df))

    path_train = 'src/guideline_tree_retrieval/data/Symptom2Disease_train.csv'
    df_new_train = pd.read_csv(path_train)
    print(len(df_new_train))
    df_new_train = remove_disease(df_new_train)
    print(len(df_new_train))


    # Split train and test
    df = df.sample(frac=1.0,random_state=1).reset_index(drop=True)
    R = 0.7
    N = int(R*len(df))
    df_train = df.loc[0:N-1]
    # df_train = df_new_train  #  modified here!!! (choose training set)
    # use df_new_train: 97%, use df_train:64%
    df_test = df.loc[N:]

    # Constrcut symptom dict for retrieval
    dict_symp2dis = {}
    for idx in range(len(df_train)):
        dict_symp2dis[df_train.iloc[idx]['text']] = df_train.iloc[idx]['label']
    # pprint(dict_symp2dis)
    symptom_keys = list(dict_symp2dis.keys())

    # Define embedding model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)
    scorer = BERTScorer(model_type='bert-base-uncased')

    cnt = 0
    err_dict = {}
    for idx in range(len(df_test)): 
        query = df_test.iloc[idx]['text']
        label_gt = df_test.iloc[idx]['label']

        # Similarity score
        score_list = []
        for idx in range(len(symptom_keys)):
            symptom = symptom_keys[idx]
            P, R, F1 = scorer.score([query], [symptom])
            score_list.append(F1.item())

        K = 1
        topk_idx = heapq.nlargest(K, range(len(score_list)), score_list.__getitem__)
        # for k in range(K):
        #     print("Disease: ", dict_symp2dis[symptom_keys[topk_idx[k]]], "  score: ", score_list[topk_idx[k]])
        label_pred = dict_symp2dis[symptom_keys[topk_idx[0]]]
        if label_gt == label_pred:
            cnt += 1
        else:
            if label_gt in err_dict:
                err_dict[label_gt] += 1
            else:
                err_dict[label_gt] = 1
        print('GT: ', label_gt, '  Pred: ', dict_symp2dis[symptom_keys[topk_idx[0]]], "  score: ", score_list[topk_idx[0]])
    print("Accuracy: ", cnt/len(df_test))
    print(err_dict)