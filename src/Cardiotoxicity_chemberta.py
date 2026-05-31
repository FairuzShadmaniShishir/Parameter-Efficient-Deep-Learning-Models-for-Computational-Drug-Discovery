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

# df_train = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_train.csv')
# df_val = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_val.csv')
# df=pd.concat([df_train,df_val])
# df['smiles'] = df['smiles_standarized'].apply(lambda s: s.replace('\n', ''))
# #test_df = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/paper_valid_data.csv')
# test_df = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_test.csv')
# test_df['smiles'] = test_df['smiles_standarized'].apply(lambda s: s.replace('\n', ''))
# #test_df['smiles'] = test_df['SMILES'].apply(lambda s: s.replace('\n', ''))
# #test_df.head()


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

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# # Create a 90-10 train-validation split.
#
# # Calculate the number of samples to include in each set.
# train_size = int(0.8 * len(dataset))
# #val_size = int(0.2 * len(dataset))
# val_size = len(dataset)  - train_size
#
# # Divide the dataset by randomly selecting samples.
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
# print('{:>5,} training samples'.format(train_size))
# print('{:>5,} validation samples'.format(val_size))
#
# batch_size = 4
# train_dataloader = DataLoader(
#             train_dataset,  # The training samples.
#             sampler = RandomSampler(train_dataset), # Select batches randomly
#             batch_size = batch_size   )
# # For validation the order doesn't matter, so we'll just read them sequentially.
# validation_dataloader = DataLoader(
#             val_dataset, # The validation samples.
#             sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
#             batch_size = batch_size # Evaluate with this batch size.
#         )
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = RobertaForSequenceClassification.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250",    num_labels = 2, # The number of output labels--2 for binary classification.
#                     # You can increase this for multi-class tasks.
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False,)
# model = model.to(device)
#
# optimizer = AdamW(model.parameters(),
#                   lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
#                 )
#
# epochs = 50
# total_steps = len(train_dataloader) * epochs
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps = 0, # Default value in run_glue.py
#                                             num_training_steps = total_steps)
#
# def flat_accuracy(preds, labels):
#     pred_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return np.sum(pred_flat == labels_flat) / len(labels_flat)
#
# def format_time(elapsed):
#     elapsed_rounded = int(round((elapsed)))
#     return str(datetime.timedelta(seconds=elapsed_rounded))
#
#
# for epoch_i in range(0, epochs):
#     print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
#     print("Training...")
#
#     t0 = time.time()
#     total_train_loss = 0
#     model.train()
#
#     for step, batch in enumerate(train_dataloader):
#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2].to(device)
#
#         optimizer.zero_grad()
#         output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#
#         # 🔍 Get logits and check shape
#         logits = output.logits  # Model output
#         # print("Logits shape:", logits.shape)  # Debugging info
#         # print("Labels shape:", b_labels.shape)
#
#         # 🛠️ Detect classification type & apply correct loss function
#         if logits.shape[1] == 1:  # Binary classification (1 logit per sample)
#             loss_fn = torch.nn.BCEWithLogitsLoss()
#             loss = loss_fn(logits.view(-1), b_labels.long())  # Flatten logits
#         else:  # Multi-class classification
#             loss_fn = torch.nn.CrossEntropyLoss()
#             loss = loss_fn(logits, b_labels.long())  # Ensure labels are long
#
#         total_train_loss += loss.item()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         scheduler.step()
#
#     avg_train_loss = total_train_loss / len(train_dataloader)
#     training_time = format_time(time.time() - t0)
#     print("\n  Average training loss: {:.2f}".format(avg_train_loss))
#     print("  Training epoch took: {:}".format(training_time))
#
#     # ======= Validation =======
#     print("\nRunning Validation...")
#     t0 = time.time()
#     model.eval()
#     total_eval_accuracy = 0
#     total_eval_loss = 0
#     total_eval_auc = 0
#     total_eval_mcc = 0
#
#     for batch in validation_dataloader:
#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2].to(device)
#
#         with torch.no_grad():
#             output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#
#         logits = output.logits
#         loss = loss_fn(logits.view(-1) if logits.shape[1] == 1 else logits,
#                        b_labels.float() if logits.shape[1] == 1 else b_labels.long())
#         total_eval_loss += loss.item()
#
#         # Move logits and labels to CPU
#         logits = logits.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()
#
#         total_eval_accuracy += flat_accuracy(logits, label_ids)
#         # total_eval_auc += roc_auc_score(label_ids, logits[:, 1] if logits.shape[1] > 1 else logits)
#         # total_eval_mcc += matthews_corrcoef(label_ids, np.round(logits[:, 1] if logits.shape[1] > 1 else logits))
#
#     avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
#     avg_val_auc = total_eval_auc / len(validation_dataloader)
#     avg_val_mcc = total_eval_mcc / len(validation_dataloader)
#     avg_val_loss = total_eval_loss / len(validation_dataloader)
#     # validation_time = format_time(time.time() - t0)
#
#     print("  Accuracy: {:.2f}".format(avg_val_accuracy))
#
# test_input_ids = []
# test_attention_masks = []
# for seq in test_df.smiles:
#     encoded_dict = tokenizer.encode_plus(
#                         seq,
#                         add_special_tokens = True,
#                         max_length = max_len,
#                         pad_to_max_length = True,
#                         return_attention_mask = True,
#                         return_tensors = 'pt',
#                    )
#     test_input_ids.append(encoded_dict['input_ids'])
#     test_attention_masks.append(encoded_dict['attention_mask'])
# test_input_ids = torch.cat(test_input_ids, dim=0)
# test_attention_masks = torch.cat(test_attention_masks, dim=0)
#
# test_dataset = TensorDataset(test_input_ids, test_attention_masks)
# test_dataloader = DataLoader(
#             test_dataset, # The validation samples.
#             sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
#             batch_size = batch_size # Evaluate with this batch size.
#         )
#
# predictions = []
# loggit = []
# for batch in test_dataloader:
#     # print(batch)
#     b_input_ids = batch[0].to(device)
#     b_input_mask = batch[1].to(device)
#     with torch.no_grad():
#         output = model(b_input_ids,
#                        token_type_ids=None,
#                        attention_mask=b_input_mask)
#         logits = output.logits
#         logits = logits.detach().cpu().numpy()
#         # print(logits)
#         probabilities = 1 / (1 + np.exp(-logits))
#         loggit.append(probabilities)
#         # Compute ROC AUC
#
#         pred_flat = np.argmax(logits, axis=1).flatten()
#         # print(pred_flat)
#         predictions.extend(list(pred_flat))
#
# df_output = pd.DataFrame()
# #df_output['id'] = df_test['id']
# df_output['target'] =predictions
#
# print(matthews_corrcoef(test_df.label, df_output['target']))
# loggit = np.vstack(loggit)
# print(roc_auc_score(test_df.label, loggit[:,1]))
# from sklearn.metrics import roc_auc_score, matthews_corrcoef, classification_report
# print(classification_report(test_df.label, df_output['target']))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report, accuracy_score, f1_score, recall_score, confusion_matrix

# Define number of folds
k_folds = 2

# Prepare dataset again
dataset = TensorDataset(input_ids, attention_masks, labels)

# Initialize KFold
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

batch_size = 4
epochs = 10   # you might want fewer epochs per fold
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# To store fold results
fold_accuracy = []
fold_mcc = []
fold_auc = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n==== Fold {fold+1} / {k_folds} ====")

    # Create dataloaders for this fold
    train_inputs = input_ids[train_idx]
    train_masks = attention_masks[train_idx]
    train_labels = labels[train_idx]

    val_inputs = input_ids[val_idx]
    val_masks = attention_masks[val_idx]
    val_labels = labels[val_idx]

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    validation_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    # Create model (reset at each fold)
    model = RobertaForSequenceClassification.from_pretrained(
        "seyonec/PubChem10M_SMILES_BPE_396_250",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=0,
                                                 num_training_steps=total_steps)

    # Training Loop
    for epoch_i in range(epochs):
        print(f"Epoch {epoch_i+1}/{epochs}")
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits

            if logits.shape[1] == 1:
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(logits.view(-1), b_labels.float())
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, b_labels.long())

            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.2f}")

    # Validation Loop
    print("Running Validation...")
    model.eval()

    preds = []
    true_labels = []

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        preds.append(logits)
        true_labels.append(label_ids)

    preds = np.vstack(preds)
    true_labels = np.concatenate(true_labels)

    if preds.shape[1] == 1:
        preds_prob = 1 / (1 + np.exp(-preds))  # sigmoid
        preds_label = (preds_prob > 0.5).astype(int).flatten()
    else:
        preds_label = np.argmax(preds, axis=1)

    # Metrics
    acc = np.mean(preds_label == true_labels)
    mcc = matthews_corrcoef(true_labels, preds_label)
    auc = roc_auc_score(true_labels, preds[:,1] if preds.shape[1] > 1 else preds.flatten())

    fold_accuracy.append(acc)
    fold_mcc.append(mcc)
    fold_auc.append(auc)

    print(f"  Fold {fold+1} Accuracy: {acc:.4f} MCC: {mcc:.4f} AUC: {auc:.4f}")

# Overall Results
print("\n==== Cross-validation Results ====")
print(f"Avg Accuracy: {np.mean(fold_accuracy):.4f} ± {np.std(fold_accuracy):.4f}")
print(f"Avg MCC: {np.mean(fold_mcc):.4f} ± {np.std(fold_mcc):.4f}")
print(f"Avg AUC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}")

import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report

# Assuming you already have `tokenizer` defined and a model loaded

# Define necessary parameters
max_len = 512  # Maximum sequence length for the model
batch_size = 4  # Batch size for evaluation

# Prepare test data by tokenizing the SMILES strings
test_input_ids = []
test_attention_masks = []

for seq in test_df.smiles:
    # Encode the SMILES strings
    encoded_dict = tokenizer.encode_plus(
        seq,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Padding/truncating to max length
        pad_to_max_length=True,  # Pad to the maximum length
        return_attention_mask=True,  # Return attention mask
        return_tensors='pt',  # Return pytorch tensors
    )

    # Append the tokenized data to the lists
    test_input_ids.append(encoded_dict['input_ids'])
    test_attention_masks.append(encoded_dict['attention_mask'])

# Convert lists to tensors
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)

# Create TensorDataset for test data
test_dataset = TensorDataset(test_input_ids, test_attention_masks)

# Create DataLoader for evaluation
test_dataloader = DataLoader(
    test_dataset,  # The test samples
    sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially
    batch_size=batch_size  # Batch size for evaluation
)

from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report, accuracy_score, f1_score, recall_score, confusion_matrix

# Initialize list to store predictions and logits
predictions = []
loggit = []

# Evaluate the model
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)

    with torch.no_grad():
        output = model(b_input_ids, attention_mask=b_input_mask)
        logits = output.logits

        logits = logits.detach().cpu().numpy()

        # For binary classification, use sigmoid to get probabilities
        probabilities = 1 / (1 + np.exp(-logits))
        loggit.append(probabilities)

        # For prediction, choose the higher probability
        pred_flat = np.argmax(logits, axis=1).flatten()
        predictions.extend(list(pred_flat))

# Prepare output DataFrame
df_output = pd.DataFrame()
df_output['target'] = predictions

# Convert logits list to array
loggit = np.vstack(loggit)

# Calculate metrics
true_labels = test_df.label.values
pred_labels = df_output['target'].values

# Basic Metrics
mcc = matthews_corrcoef(true_labels, pred_labels)
auc = roc_auc_score(true_labels, loggit[:, 1])
acc = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
sensitivity = recall_score(true_labels, pred_labels)  # recall for positive class

# Specificity (recall for negative class)
tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
specificity = tn / (tn + fp)

# Print results
print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
print(f'ROC AUC: {auc:.4f}')
print(f'Accuracy: {acc:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Sensitivity (Recall for Positive Class): {sensitivity:.4f}')
print(f'Specificity (Recall for Negative Class): {specificity:.4f}')
print()
print(classification_report(true_labels, pred_labels, digits=4))

# Calculate total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable ChemBERT parameters: {total_params:,}")


# import math
#
# class LoRALayer(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, rank, alpha):
#         super().__init__()
#         self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
#         torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
#         self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
#         self.alpha = alpha
#
#     def forward(self, x):
#         x = self.alpha * (x @ self.A @ self.B)
#         return x
#
# class LinearWithLoRA(torch.nn.Module):
#     def __init__(self, linear, rank, alpha):
#         super().__init__()
#         self.linear = linear
#         self.lora = LoRALayer(
#             linear.in_features, linear.out_features, rank, alpha
#         )
#
#     def forward(self, x):
#         return self.linear(x) + self.lora(x)
#
# def replace_linear_with_lora(model, rank, alpha):
#     for name, module in model.named_children():
#         if isinstance(module, torch.nn.Linear):
#             # Replace the Linear layer with LinearWithLoRA
#             setattr(model, name, LinearWithLoRA(module, rank, alpha))
#         else:
#             # Recursively apply the same function to child modules
#             replace_linear_with_lora(module, rank, alpha)
#
# replace_linear_with_lora(model, rank=16, alpha=16)
# model = model.to(device)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total trainable LoRA parameters: {total_params:,}")
#
#
#
# from sklearn.metrics import matthews_corrcoef
# from sklearn.metrics import roc_auc_score
#
# for epoch_i in range(0, epochs):
#     print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
#     print("Training...")
#
#     t0 = time.time()
#     total_train_loss = 0
#     model.train()
#
#     for step, batch in enumerate(train_dataloader):
#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2].to(device)
#
#         optimizer.zero_grad()
#         output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#
#         # 🔍 Get logits and check shape
#         logits = output.logits  # Model output
#         # print("Logits shape:", logits.shape)  # Debugging info
#         # print("Labels shape:", b_labels.shape)
#
#         # 🛠️ Detect classification type & apply correct loss function
#         if logits.shape[1] == 1:  # Binary classification (1 logit per sample)
#             loss_fn = torch.nn.BCEWithLogitsLoss()
#             loss = loss_fn(logits.view(-1), b_labels.float())  # Flatten logits
#         else:  # Multi-class classification
#             loss_fn = torch.nn.CrossEntropyLoss()
#             loss = loss_fn(logits, b_labels.long())  # Ensure labels are long
#
#         total_train_loss += loss.item()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         scheduler.step()
#
#     avg_train_loss = total_train_loss / len(train_dataloader)
#     training_time = format_time(time.time() - t0)
#     print("\n  Average training loss: {:.2f}".format(avg_train_loss))
#     print("  Training epoch took: {:}".format(training_time))
#
#     # ======= Validation =======
#     print("\nRunning Validation...")
#     t0 = time.time()
#     model.eval()
#     total_eval_accuracy = 0
#     total_eval_loss = 0
#     total_eval_auc = 0
#     total_eval_mcc = 0
#
#     for batch in validation_dataloader:
#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2].to(device)
#
#         with torch.no_grad():
#             output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#
#         logits = output.logits
#         loss = loss_fn(logits.view(-1) if logits.shape[1] == 1 else logits,
#                        b_labels.float() if logits.shape[1] == 1 else b_labels.long())
#         total_eval_loss += loss.item()
#
#         # Move logits and labels to CPU
#         logits = logits.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()
#
#         total_eval_accuracy += flat_accuracy(logits, label_ids)
#         # total_eval_auc += roc_auc_score(label_ids, logits[:, 1] if logits.shape[1] > 1 else logits)
#         # total_eval_mcc += matthews_corrcoef(label_ids, np.round(logits[:, 1] if logits.shape[1] > 1 else logits))
#
#     avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
#     avg_val_auc = total_eval_auc / len(validation_dataloader)
#     avg_val_mcc = total_eval_mcc / len(validation_dataloader)
#     avg_val_loss = total_eval_loss / len(validation_dataloader)
#     # validation_time = format_time(time.time() - t0)
#
#     print("  Accuracy: {:.2f}".format(avg_val_accuracy))
#
# predictions = []
# loggit = []
# for batch in test_dataloader:
#     # print(batch)
#     b_input_ids = batch[0].to(device)
#     b_input_mask = batch[1].to(device)
#     with torch.no_grad():
#         output = model(b_input_ids,
#                        token_type_ids=None,
#                        attention_mask=b_input_mask)
#         logits = output.logits
#         logits = logits.detach().cpu().numpy()
#         # print(logits)
#         probabilities = 1 / (1 + np.exp(-logits))
#         loggit.append(probabilities)
#         # Compute ROC AUC
#
#         pred_flat = np.argmax(logits, axis=1).flatten()
#         # print(pred_flat)
#         predictions.extend(list(pred_flat))
#
# df_output = pd.DataFrame()
# #df_output['id'] = df_test['id']
# df_output['target'] =predictions
# print(matthews_corrcoef(test_df.label, df_output['target']))
# loggit = np.vstack(loggit)
# print(roc_auc_score(test_df.label, loggit[:,1]))
# #print(loggit[:,1])
#
# from sklearn.metrics import confusion_matrix,classification_report
# print(classification_report(test_df.label, df_output['target']))
