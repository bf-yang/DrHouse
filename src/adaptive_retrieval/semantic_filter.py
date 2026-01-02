# Peformance of semantic filter (SLM) on our collected dataset
# Task is to determine whether to retrieve sensor data (classification task)
import pandas as pd
import os, random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from transformers import  BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
from huggingface_hub import snapshot_download
from transformers import BertTokenizer
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(save_path, save_name, model):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


class BertClassifier(nn.Module):
    def __init__(self, flag):
        super(BertClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained("bert-mini-cased")
        # self.bert = BertModel.from_pretrained("bert-medium-uncased")
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        # self.bert = BertModel.from_pretrained("bert-large-uncased")
        self.flag_bert = flag
        self.dropout = nn.Dropout(0.5)

        if self.flag_bert == "base":
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.linear = nn.Linear(768, 2)    
        elif self.flag_bert == "large":
            self.bert = BertModel.from_pretrained("bert-large-uncased")
            self.linear = nn.Linear(1024, 2)
        elif self.flag_bert == "medium":
            self.bert = BertModel.from_pretrained("google/bert_uncased_L-8_H-512_A-8")
            self.linear = nn.Linear(512, 2)
        elif self.flag_bert == "small":
            self.bert = BertModel.from_pretrained("prajjwal1/bert-small")
            self.linear = nn.Linear(512, 2)
        elif self.flag_bert == "tiny":
            self.bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
            self.linear = nn.Linear(128, 2)
        elif self.flag_bert == "Roberta":
            self.bert = RobertaModel.from_pretrained("roberta-base")
            self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
    

class MyDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.df = df
        self.length_counts = [len(item.split(" ")) for item in self.df['text']]
        # print(max(self.length_counts))
        self.texts = [self.tokenizer(text, 
                                padding='max_length',
                                max_length = max(self.length_counts), 
                                truncation=True,
                                return_tensors="pt") 
                      for text in df['text']]
        self.labels = [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
    


if __name__ == "__main__":
    best_dev_acc = 0
    epoch = 25
    batch_size = 32
    lr = 1e-5
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    random_seed = 1999
    save_path = 'src/adaptive_retrieval/checkpoints'
    setup_seed(random_seed)

    # Load training set and test set
    train_df = pd.read_excel("src/adaptive_retrieval/data/train.xlsx", sheet_name='dataset')
    test_df = pd.read_excel("src/adaptive_retrieval/data/test.xlsx", sheet_name='dataset')

    # length_counts = [len(item.split(" ")) for item in df['text']]
    # print(max(length_counts))

    # # Split train and test
    # df = df.sample(frac=1.0, random_state=42).reset_index()
    # train_df = df.sample(frac=0.7, random_state=42)
    # test_df = df[~df.index.isin(train_df.index)]

    # tokenizer = BertTokenizer.from_pretrained('bert-mini-cased')
    # tokenizer = BertTokenizer.from_pretrained('bert-medium-uncased')


    # flag_bert = "tiny"
    # flag_bert = "small"
    # flag_bert = "medium"
    flag_bert = "base"
    # flag_bert = "large"
    # flag_bert = "Roberta"
    

    model = BertClassifier(flag=flag_bert)

    if flag_bert == "base":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif flag_bert == "large":
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    elif flag_bert == "medium":
        tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-8_H-512_A-8') 
    elif flag_bert == "small":
        tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small') 
    elif flag_bert == "tiny":
        tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny') 
    elif flag_bert == "Roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base") 

    # Dataloader
    train_dataset = MyDataset(train_df,tokenizer)
    dev_dataset = MyDataset(test_df,tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # Training setting
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
    model = model.to(device)
    criterion = criterion.to(device)

    # Training and testing
    for epoch_num in range(epoch):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            masks = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += batch_loss.item()

    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    with torch.no_grad(): 
        
        flag_test_time = 1
        tmp = []
        if flag_test_time:
            times = 100
        else:
            times = 1
        for idx in range(times):
            for inputs, labels in dev_loader:
                input_ids = inputs['input_ids'].squeeze(1).to(device) 
                masks = inputs['attention_mask'].to(device) 
                labels = labels.to(device)
                start = time.time()
                output = model(input_ids, masks)
                end = time.time()
                tmp.append(end-start)

                batch_loss = criterion(output, labels)            
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc
                total_loss_val += batch_loss.item()    
        if flag_test_time:
            count_time = np.sum(tmp[50:])/50
        else:
            count_time = 0
        print(f'''Epochs: {epoch_num + 1} 
            | Train Loss: {total_loss_train / len(train_dataset): .3f} 
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
            | Val Loss: {total_loss_val / len(dev_dataset): .3f} 
            | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}
            | Time"  { count_time:.6f} )
            '''
            )
        
        # Save best weights
        if total_acc_val / len(dev_dataset) > best_dev_acc:
            best_dev_acc = total_acc_val / len(dev_dataset)
            save_model(save_path, 'weights_best.pt', model)
            print("End Training")