from tqdm import tqdm
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import Draw
from rdkit import Chem
RDLogger.DisableLog("rdApp.*")
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, matthews_corrcoef, classification_report
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import GINConv
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc

df_train = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_train.csv')
df_val = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_val.csv')
df=pd.concat([df_train,df_val])
df['smiles'] = df['smiles_standarized'].apply(lambda s: s.replace('\n', ''))
test_df = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/paper_valid_data.csv')
#test_df['smiles'] = test_df['smiles_standarized'].apply(lambda s: s.replace('\n', ''))
test_df['smiles'] = test_df['SMILES'].apply(lambda s: s.replace('\n', ''))


def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, use_chirality=True):
    # Define a simplified list of atom types
    permitted_atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Dy', 'Unknown']
    atom_type = atom.GetSymbol() if atom.GetSymbol() in permitted_atom_types else 'Unknown'
    atom_type_enc = one_hot_encoding(atom_type, permitted_atom_types)

    # Consider only the most impactful features: atom degree and whether the atom is in a ring
    atom_degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 'MoreThanFour'])
    is_in_ring = [int(atom.IsInRing())]

    # Optionally include chirality
    if use_chirality:
        chirality_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                         ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_features = atom_type_enc + atom_degree + is_in_ring + chirality_enc
    else:
        atom_features = atom_type_enc + atom_degree + is_in_ring

    return np.array(atom_features, dtype=np.float32)

def get_bond_features(bond):
    # Simplified list of bond types
    permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                            Chem.rdchem.BondType.AROMATIC, 'Unknown']
    bond_type = bond.GetBondType() if bond.GetBondType() in permitted_bond_types else 'Unknown'

    # Features: Bond type, Is in a ring
    features = one_hot_encoding(bond_type, permitted_bond_types) \
               + [int(bond.IsInRing())]

    return np.array(features, dtype=np.float32)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    data_list = []

    for index, smiles in enumerate(x_smiles):
        mol = Chem.MolFromSmiles(smiles)

        if not mol:  # Skip invalid SMILES strings
            continue

        # Node features
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge features
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [(start, end), (end, start)]  # Undirected graph
            bond_feature = get_bond_features(bond)
            edge_features += [bond_feature, bond_feature]  # Same features in both directions

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Creating the Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.y = torch.tensor(y[index], dtype=torch.float)
        data_list.append(data)

    return data_list

def featurize_data_in_batches(smiles_list, labels_list, batch_size):
    data_list = []
    # Define tqdm progress bar
    pbar = tqdm(total=len(smiles_list), desc="Featurizing data")
    for i in range(0, len(smiles_list), batch_size):
        smiles_batch = smiles_list[i:i + batch_size]
        labels_batch = labels_list[i:i + batch_size]

        batch_data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles_batch,labels_batch)
        data_list.extend(batch_data_list)
        pbar.update(len(smiles_batch))

    pbar.close()
    return data_list

batch_size = 2
featurized_data = featurize_data_in_batches(df.smiles.tolist(),df.label.tolist(), batch_size)
test_data = featurize_data_in_batches(test_df.smiles.tolist(), test_df.label.tolist(), batch_size)

class CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGNNLayer, self).__init__(aggr='max')
        self.lin = nn.Linear(in_channels + 6, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages
        return MessagePassing.propagate(self, edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        combined = torch.cat((x_j, edge_attr), dim=1)
        return combined

    def update(self, aggr_out):
        return self.lin(aggr_out)

#Define GNN Model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList([CustomGNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.gelu(x)
            x = self.dropout(x)


        x = global_max_pool(x, data.batch) # Global pooling to get a graph-level representation
        x = self.lin(x)
        return x


def train_model(train_loader, num_epochs, input_dim, hidden_dim, num_layers, dropout_rate, lr):
    # GNN model
    model = GNNModel(input_dim, hidden_dim, num_layers, dropout_rate)
    # model=GATRoPEModel(input_dim, hidden_dim,out_dim, num_layers, edge_dim, num_heads, dropout_rate)
    # GIN Model
    # model = GIN(hidden_dim, num_layers,dropout_rate)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

    return model


def predict_with_model(model, test_loader):
    model.eval()
    predictions = []
    # molecule_ids = []

    with torch.no_grad():
        for data in test_loader:
            output = torch.sigmoid(model(data))
            predictions.extend(output.view(-1).tolist())
            # molecule_ids.extend(data.molecule_id)

    return predictions


from torch_geometric.loader import DataLoader

# Create DataLoaders for the current protein
train_loader = DataLoader(featurized_data, batch_size=2, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=False, drop_last=False)

# Train model
input_dim = train_loader.dataset[0].num_node_features
hidden_dim = 64
num_epochs = 100
num_layers = 4  # Should ideally be set so that all nodes can communicate with each other
dropout_rate = 0.1
lr = 0.0001
# edge_dim=6
# These are just example values, feel free to play around with them.
# tracker = EmissionsTracker(project_name="pytorch-carbon-tracking")
# tracker.start()
model = train_model(train_loader, num_epochs, input_dim, hidden_dim, num_layers, dropout_rate, lr)
# emissions = tracker.stop()

# Predict
custom_predictions = predict_with_model(model, test_loader)

# plt.figure(figsize=(7, 5))
# fpr, tpr, _ = roc_curve(test_df['label'], custom_predictions)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, label = f'{'GNN Model'} (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
#
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves')
# plt.legend()
# plt.show()
auc = roc_auc_score(test_df['label'], custom_predictions)
mcc = matthews_corrcoef(test_df['label'], [round(i) for i in custom_predictions])
print(f'AUC: {auc:.4f}')
print(f'MCC: {mcc:.4f}')
print("\nClassification Report:")
print(classification_report(test_df['label'], [round(i) for i in custom_predictions]))