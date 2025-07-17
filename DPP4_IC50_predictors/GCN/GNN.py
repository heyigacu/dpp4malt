import os
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_fold_indices(fold_number):
    train_index, test_index = np.load(f'{root_dir}/dataset/splits/scaffold-{fold_number}.npy', allow_pickle=True)
    return train_index, test_index

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    hybridizations = [Chem.rdchem.HybridizationType.SP,
                      Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3]

    x = []
    for atom in mol.GetAtoms():
        features = []

        # One-hot atom type
        atom_type = atom.GetSymbol()
        features.extend([int(atom_type == t) for t in atom_types])

        # Degree (0–5)
        degree = atom.GetDegree()
        features.extend([int(degree == i) for i in range(6)])

        # Total Hs
        num_h = atom.GetTotalNumHs()
        features.extend([int(num_h == i) for i in range(5)])

        # In ring
        features.append(int(atom.IsInRing()))

        # Aromatic
        features.append(int(atom.GetIsAromatic()))

        # Hybridization
        hyb = atom.GetHybridization()
        features.extend([int(hyb == h) for h in hybridizations])

        x.append(features)

    x = torch.tensor(x, dtype=torch.float)

    # Edge index
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = Sequential(
            Linear(hidden_channels, 64),
            ReLU(),
            Linear(64, 1)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()

def QSAR_with_GNN(smiles_list, labels, save_dir):
    graphs = []
    valid_labels = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        g = mol_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_labels.append(labels[i])
            valid_indices.append(i)

    for fold in range(5):
        train_idx, test_idx = read_fold_indices(fold)
        train_idx = [i for i in train_idx if i in valid_indices]
        test_idx = [i for i in test_idx if i in valid_indices]

        train_graphs = [graphs[valid_indices.index(i)] for i in train_idx]
        test_graphs = [graphs[valid_indices.index(i)] for i in test_idx]
        y_train = torch.tensor([labels[i] for i in train_idx], dtype=torch.float)
        y_test = torch.tensor([labels[i] for i in test_idx], dtype=torch.float)

        for i, g in enumerate(train_graphs):
            g.y = y_train[i:i+1]
        for i, g in enumerate(test_graphs):
            g.y = y_test[i:i+1]

        train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=128)

        in_channels = train_graphs[0].x.shape[1]
        model = GCN(in_channels=in_channels, hidden_channels=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = MSELoss()
        best_loss = float('inf')
        counter = 0
        patience = 40

        for epoch in range(1000):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(out, batch.y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            # Eval
            model.eval()
            with torch.no_grad():
                y_true, y_pred = [], []
                for batch in test_loader:
                    out = model(batch.x, batch.edge_index, batch.batch)
                    y_true.append(batch.y.view(-1))
                    y_pred.append(out)
                y_true = torch.cat(y_true)
                y_pred = torch.cat(y_pred)
                val_loss = loss_fn(y_pred, y_true).item()

            print(f"Fold {fold}, Epoch {epoch}, TrainLoss {avg_loss:.4f}, TestLoss {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f'fold-{fold}-best.pt'))
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}. Best epoch: {epoch - counter}")
                    break

        # Predict and save
        model.load_state_dict(torch.load(os.path.join(save_dir, f'fold-{fold}-best.pt')))
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                y_true.append(batch.y.view(-1))
                y_pred.append(out)
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)

            with open(os.path.join(save_dir, f'fold-{fold}-test.txt'), 'w') as f:
                for a, p in zip(y_true, y_pred):
                    f.write(f'{a.item()} {p.item()}\n')


def predict_external_dataset(smiles_list, model_dir, save_path=None):
    graphs = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        g = mol_to_graph(smi)
        print(g)
        if g is not None:
            graphs.append(g)
            valid_indices.append(i)

    loader = DataLoader(graphs, batch_size=64, shuffle=False)

    predictions = []

    for fold in range(5):
        model = GCN(in_channels=25, hidden_channels=128)
        model.load_state_dict(torch.load(os.path.join(model_dir, f'fold-{fold}-best.pt')))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                fold_preds.extend(out.cpu().numpy())
        predictions.append(fold_preds)

    preds_array = torch.tensor(predictions)
    avg_preds = preds_array.mean(dim=0).numpy()


    if save_path:
        with open(save_path, 'w') as f:
            for smi, pred in zip([smiles_list[i] for i in valid_indices], avg_preds):
                f.write(f'{smi}\t{pred:.4f}\n')
        print(f"[✓] Predictions saved to {save_path}")

    return avg_preds

if __name__ == "__main__":
    # input_file_path = root_dir + '/dataset/clean_dpp4_ic50.csv'
    save_dir = root_dir + '/GCN/'

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # df = pd.read_csv(input_file_path, sep='\t')
    # smiles = df['Smiles'].tolist()
    # labels = df['Label'].tolist()

    # QSAR_with_GNN(smiles, labels, save_dir)
    input_file_path = root_dir + '/dataset/clean_external_test.csv'
    df = pd.read_csv(input_file_path, sep='\t')
    smiles = df['Smiles'].tolist()
    labels = df['pIC50'].tolist()
    predict_external_dataset(smiles, save_dir, save_path=save_dir+'test_result')