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
RDLogger.DisableLog("rdApp.*")

df_train = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_train.csv')
df_val = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_val.csv')
df=pd.concat([df_train,df_val])
df['smiles'] = df['smiles_standarized'].apply(lambda s: s.replace('\n', ''))
#test_df = pd.read_csv('/home/f087s426/Research/Masters_Thesis/herg/herg_test.csv')
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
test_Mol_descriptors, desc_names = RDkit_descriptors(test_df['smiles'])
df_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
testdf_with_200_descriptors = pd.DataFrame(test_Mol_descriptors,columns=desc_names)
scaler = StandardScaler()
df_with_200_descriptors = scaler.fit_transform(df_with_200_descriptors)
testdf_with_200_descriptors = scaler.fit_transform(testdf_with_200_descriptors)
xgb_model = xgb.XGBClassifier(missing=np.nan,eval_metric='logloss')
xgb_model.fit(df_with_200_descriptors,df.label)
y_pred = xgb_model.predict_proba(testdf_with_200_descriptors)
y_pred_class = (y_pred[:, 1] > 0.7).astype(int)
print(y_pred[:, 1])
auc = roc_auc_score(test_df['label'], y_pred[:, 1])
mcc = matthews_corrcoef(test_df['label'], y_pred_class)

print(f'AUC: {auc:.4f}')
print(f'MCC: {mcc:.4f}')
print("\nClassification Report:")
print(classification_report(test_df['label'], y_pred_class))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(test_df['label'], y_pred_class)

# Print confusion matrix in text form
print("Confusion Matrix:")
print(cm)