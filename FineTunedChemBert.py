import ast
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import xgboost as xgb
from sklearn.metrics import roc_auc_score, matthews_corrcoef, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from mordred import Calculator, descriptors
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import Draw

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
import numpy as np
import pandas as pd
import time
import datetime
import gc
import random
#from nltk.corpus import stopwords
import re
from transformers import RobertaForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")

df_train = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_train.csv')
df_val = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_val.csv')
df_test = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_test.csv')
# herg_data = pd.read_csv("/home/f087s426/Research/Masters_Thesis/herg/hERG_IC50.csv")
# herg_data=herg_data[['Smiles','pChEMBL Value']]
# herg_data=herg_data.dropna()
# herg_data['label'] = herg_data['pChEMBL Value'].apply(lambda x: 1 if x >= 6.5 else 0)
# herg_data['group']= 1
# herg_data['smiles_standarized']=herg_data['Smiles']
# herg_data=herg_data[['smiles_standarized','label','group']]


df=pd.concat([df_train,df_val,df_test])
df['smiles'] = df['smiles_standarized'].apply(lambda s: s.replace('\n', ''))
print(len(df))

test_df = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/paper_valid_data.csv')
#test_df['smiles'] = test_df['smiles_standarized'].apply(lambda s: s.replace('\n', ''))
test_df['smiles'] = test_df['SMILES'].apply(lambda s: s.replace('\n', ''))
test_df.head()


def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    Mol_descriptors = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors, desc_names


Mol_descriptors, desc_names = RDkit_descriptors(df['smiles'])
df_with_200_descriptors = pd.DataFrame(Mol_descriptors, columns=desc_names)
#df_with_200_descriptors = df_with_200_descriptors.select_dtypes(include=["number"])
test_Mol_descriptors, desc_names = RDkit_descriptors(test_df['smiles'])
testdf_with_200_descriptors = pd.DataFrame(test_Mol_descriptors, columns=desc_names)
#testdf_with_200_descriptors = testdf_with_200_descriptors.select_dtypes(include=["number"])


# common_cols = df_with_200_descriptors.columns.intersection(testdf_with_200_descriptors.columns)
# df_with_200_descriptors = df_with_200_descriptors[common_cols]
# testdf_with_200_descriptors = testdf_with_200_descriptors[common_cols]

print(df_with_200_descriptors.values.shape)
print(testdf_with_200_descriptors.values.shape)

# scaler = StandardScaler()
# df_with_200_descriptors = scaler.fit_transform(df_with_200_descriptors)
# testdf_with_200_descriptors = scaler.transform(testdf_with_200_descriptors)

tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250")
max_len = 0
for seq in df.smiles:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(seq, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

#print('Max sentence length: ', max_len)

input_ids = []
attention_masks = []

# For every tweet...
for seq in df.smiles:
    encoded_dict = tokenizer.encode_plus(
        seq,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df.label.tolist())

# Print sentence 0, now as a list of IDs.
#print('Original: ', df.smiles[0])
#print('Token IDs:', input_ids[0])
numerical_features=df_with_200_descriptors
dataset = TensorDataset(input_ids, attention_masks, torch.tensor(numerical_features.values,dtype=torch.float32), labels)

train_size = int(0.8 * len(dataset))
#val_size = int(0.2 * len(dataset))
val_size = len(dataset)  - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

batch_size = 4
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size   )
# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for step, batch in enumerate(train_dataloader):
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_num_feats = batch[2].to(device)  # numerical features
    b_labels = batch[3].to(device)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import RobertaModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


# Define focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Define Cross-Attention Module
class CrossAttentionFusion(nn.Module):
    def __init__(self, bert_dim, num_dim, hidden_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(num_dim, hidden_dim)
        self.key_proj = nn.Linear(bert_dim, hidden_dim)
        self.value_proj = nn.Linear(bert_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, bert_out, num_feats):
        Q = self.query_proj(num_feats).unsqueeze(1)
        K = self.key_proj(bert_out).unsqueeze(1)
        V = self.value_proj(bert_out).unsqueeze(1)
        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5), dim=-1)
        attn_output = torch.bmm(attn_weights, V).squeeze(1)
        return self.out_proj(attn_output)

# Define Full Model
class RobertaWithCrossAttention(nn.Module):
    def __init__(self, num_numerical_feats, num_labels):
        super(RobertaWithCrossAttention, self).__init__()
        self.roberta = RobertaModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250")
        self.dropout = nn.Dropout(0.2)
        self.num_feats_proj = nn.Sequential(
            nn.Linear(num_numerical_feats, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.cross_attention = CrossAttentionFusion(
            bert_dim=self.roberta.config.hidden_size,
            num_dim=64,
            hidden_dim=256
        )
        self.ln = LayerNorm(256)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, numerical_feats):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = roberta_outputs.pooler_output
        num_feats = self.num_feats_proj(numerical_feats)
        fused = self.cross_attention(pooled_output, num_feats)
        fused = self.ln(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

# Train
model = RobertaWithCrossAttention(num_numerical_feats=df_with_200_descriptors.shape[1], num_labels=2).to(device)
loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 3

for epoch in range(10):
    print(f"\nEpoch {epoch + 1}/10")
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_num_feats = batch[2].to(device).float()
        b_labels = batch[3].to(device).long()

        model.zero_grad()
        logits = model(b_input_ids, b_input_mask, b_num_feats)
        loss = loss_fn(logits, b_labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_num_feats = batch[2].to(device).float()
            b_labels = batch[3].to(device).long()
            logits = model(b_input_ids, b_input_mask, b_num_feats)
            loss = loss_fn(logits, b_labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(validation_dataloader)
    print(f"Validation loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("⏹️ Early stopping triggered.")
            break

from torch.utils.data import TensorDataset, DataLoader

test_numerical_feats = testdf_with_200_descriptors

test_input_ids = []
test_attention_masks = []
for seq in test_df.smiles:
    encoded_dict = tokenizer.encode_plus(
        seq,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    test_input_ids.append(encoded_dict['input_ids'])

    test_attention_masks.append(encoded_dict['attention_mask'])

test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_dataset = TensorDataset(
    test_input_ids,
    test_attention_masks,
    torch.tensor(test_numerical_feats, dtype=torch.float32),
    torch.tensor(test_df.label, dtype=torch.long)  # only if labels exist
)

test_dataloader = DataLoader(test_dataset, batch_size=64)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_num_feats = batch[2].to(device)
        b_labels = batch[3].to(device)

        logits = model(b_input_ids, b_input_mask, b_num_feats)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())



print("Accuracy:", accuracy_score(all_labels, all_preds))
print("F1 Score:", f1_score(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds))

with torch.no_grad():
    probs = []
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_num_feats = batch[2].to(device)

        logits = model(b_input_ids, b_input_mask, b_num_feats)
        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

roc_auc = roc_auc_score(all_labels, probs)
print("ROC AUC:", roc_auc)