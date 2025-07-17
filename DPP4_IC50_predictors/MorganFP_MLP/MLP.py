import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib  # 用于模型保存/加载

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_fold_indices(fold_number):
    train_index, test_index = np.load(f'{root_dir}/dataset/splits/scaffold-{fold_number}.npy', allow_pickle=True)
    return train_index, test_index

def smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    fingerprints = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,))
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
            valid_indices.append(i)
    return np.array(fingerprints), valid_indices

def QSAR_with_Morgan_MLP(smiles, labels, save_dir):
    X_all, valid_indices = smiles_to_morgan(smiles)
    y_all = np.array(labels)[valid_indices]

    for fold in range(5):
        train_index, test_index = read_fold_indices(fold)
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        mlp = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Fold {fold + 1}, MSE: {mse:.4f}")

        y_train_pred = mlp.predict(X_train)

        # 保存预测结果
        with open(os.path.join(save_dir, f'fold-{fold}-test.txt'), 'w') as f:
            for actual, predicted in zip(y_test, y_pred):
                f.write(f'{actual} {predicted}\n')
        with open(os.path.join(save_dir, f'fold-{fold}-train.txt'), 'w') as f:
            for actual, predicted in zip(y_train, y_train_pred):
                f.write(f'{actual} {predicted}\n')

        # 保存模型和Scaler
        joblib.dump(mlp, os.path.join(save_dir, f'mlp_fold_{fold}.pkl'))
        joblib.dump(scaler, os.path.join(save_dir, f'scaler_fold_{fold}.pkl'))

def predict_average_from_smiles(smiles_list, save_dir, output_file):
    """
    使用5折模型对新smiles预测，并保存平均预测值。
    """
    all_preds = []
    valid_smiles = None

    for fold in range(5):
        model_path = os.path.join(save_dir, f'mlp_fold_{fold}.pkl')
        scaler_path = os.path.join(save_dir, f'scaler_fold_{fold}.pkl')

        # 加载模型和标准化器
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # 转化指纹并记录第一次的有效 SMILES
        X, valid_indices = smiles_to_morgan(smiles_list)
        if len(X) == 0:
            raise ValueError("No valid SMILES found.")

        if valid_smiles is None:
            valid_smiles = [smiles_list[i] for i in valid_indices]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        all_preds.append(y_pred)

    # 求平均
    mean_preds = np.mean(np.vstack(all_preds), axis=0)

    # 保存为制表符分隔的文件
    with open(output_file, 'w') as f:
        f.write("SMILES\tPredicted_IC50\n")
        for smi, pred in zip(valid_smiles, mean_preds):
            f.write(f"{smi}\t{pred:.4f}\n")



if __name__ == "__main__":
    input_file_path = root_dir + '/dataset/clean_dpp4_ic50.csv'
    save_dir = root_dir + '/MorganFP_MLP/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv(input_file_path, sep='\t')
    smiles = df['Smiles']
    labels = df['Label']

    QSAR_with_Morgan_MLP(smiles, labels, save_dir)


    input_file_path = root_dir + '/dataset/clean_external_test.csv'
    df = pd.read_csv(input_file_path, sep='\t')
    test_smiles = df['Smiles'].tolist()
    labels = df['pIC50'].tolist()
    model_file = os.path.join(save_dir, 'mlp_fold_0.pkl')
    scaler_file = os.path.join(save_dir, 'scaler_fold_0.pkl')
    predict_average_from_smiles(test_smiles, save_dir, save_dir+'test.txt')
