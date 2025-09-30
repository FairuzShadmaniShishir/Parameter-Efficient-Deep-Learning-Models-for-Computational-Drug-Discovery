from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np


# Atom Featurisation
## Auxiliary function for one-hot enconding transformation based on list of
##permitted values

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


# Main atom feat. func

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


# Bond featurization

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
        # data.molecule_id = ids[index]

        # data.y = torch.tensor([y[index]], dtype=torch.float)

        data_list.append(data)

    return data_list


from tqdm import tqdm


def featurize_data_in_batches(smiles_list, labels_list, batch_size):
    data_list = []
    # Define tqdm progress bar
    pbar = tqdm(total=len(smiles_list), desc="Featurizing data")
    for i in range(0, len(smiles_list), batch_size):
        smiles_batch = smiles_list[i:i + batch_size]
        labels_batch = labels_list[i:i + batch_size]
        # ids_batch = ids_list[i:i+batch_size]
        batch_data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles_batch, labels_batch)
        data_list.extend(batch_data_list)
        pbar.update(len(smiles_batch))

    pbar.close()
    return data_list


import pandas as  pd
DAVIS_test=pd.read_csv('/home/f087s426/Research/Masters_Thesis/leash_BELKA/DAVIS.csv')
#KIBA_test=pd.read_csv('/home/f087s426/Research/Masters_Thesis/leash_BELKA/KIBA.csv')
#Binding_DB_KD_test=pd.read_csv('/home/f087s426/Research/Masters_Thesis/leash_BELKA/Binding_DB_KD.csv')

import numpy as np

# Convert Kd to pKd: pKd = -log10(Kd * 1e-9)
DAVIS_test['pKd'] = -np.log10(DAVIS_test['Y'] * 1e-9)

# # Binarize using a threshold
threshold = 7.0  # e.g., pKd > 7 means strong binding
DAVIS_test['Y_binary'] = (DAVIS_test['pKd'] > threshold).astype(int)

batch_size = 8
# List of proteins and their corresponding dataframes
# proteins_data = {
#     'sEH': seh_train_df,
#     'BRD4': brd4_train_df,
#     'HSA': hsa_train_df
# }
# Dictionary to store the featurized data for each protein
#featurized_data = {}
# Loop over each protein and its dataframe
#for protein_name, df in proteins_data.items():
    #print(f"Processing {protein_name}...")
smiles_list = DAVIS_test['Drug'].tolist()
#ids_list = df['id'].tolist()
labels_list = DAVIS_test['Y'].tolist()
# Featurize the data
featurized_data = featurize_data_in_batches(smiles_list, labels_list, batch_size)

test_loader = DataLoader(featurized_data, batch_size=32, shuffle=False)

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np


# Define custom GNN layer
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


# Define GNN Model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList(
            [CustomGNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_max_pool(x, data.batch)  # Global pooling to get a graph-level representation
        x = self.lin(x)
        return x

input_dim = test_loader.dataset[0].num_node_features
hidden_dim = 128
num_epochs = 50
num_layers = 4 #Should ideally be set so that all nodes can communicate with each other
dropout_rate = 0.3
lr = 0.001
#These are just example values, feel free to play around with them.
model = GNNModel(input_dim, hidden_dim, num_layers, dropout_rate)

#model = train_model(train_loader,num_epochs, input_dim, hidden_dim,num_layers, dropout_rate, lr)
model.load_state_dict(torch.load('/home/f087s426/jupyter notebook/gnn_model.pth'))
model.eval()
def predict_with_model(model, test_loader):
    model.eval()
    predictions = []
    #molecule_ids = []

    with torch.no_grad():
        for data in test_loader:
            output = torch.sigmoid(model(data))
            predictions.extend(output.view(-1).tolist())
            #molecule_ids.extend(data.molecule_id)

    return predictions


predictions = predict_with_model(model, test_loader)

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Get precision, recall, and thresholds
precision, recall, _ = precision_recall_curve(DAVIS_test.Y_binary, predictions)

# Compute AUPR
aupr = average_precision_score(DAVIS_test.Y_binary, predictions)

# Plot the PR curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'GNN Model on DAVIS (AUPR = {aupr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()