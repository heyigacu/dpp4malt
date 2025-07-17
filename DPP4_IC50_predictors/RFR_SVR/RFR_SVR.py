import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_fold_indices(fold_number):
    train_index, test_index = np.load(f'{root_dir}/dataset/splits/scaffold-{fold_number}.npy', allow_pickle=True)
    return train_index, test_index

def multicollinearity_analysis(descriptors_save_path, test_features_path='dataset/test_descriptors.csv'):
    df = pd.read_csv(descriptors_save_path, sep='\t')
    if 'DPP_IV_Activity' in df.columns:
        df = df.drop(columns=['DPP_IV_Activity'])
    df = df.fillna(df.mean())

    df_test = pd.read_csv(test_features_path, sep='\t')
    if 'DPP_IV_Activity' in df_test.columns:
        df_test = df_test.drop(columns=['DPP_IV_Activity'])
    df_test = df_test.fillna(df_test.mean())

    correlation_matrix = df.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper_triangle.columns:
        if col not in to_drop:
            correlated_columns = [other_col for other_col in upper_triangle.index if upper_triangle.at[other_col, col] > 0.9]
            to_drop.update(correlated_columns)

    df_reduced = df.drop(columns=list(to_drop))
    df_test_reduced = df_test[df_reduced.columns]

    return df_reduced, df_test_reduced

def feature_selection_by_random_forest(X_train, y_train, n_estimators=100):
    """
    Feature selection using RandomForest feature importance.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    feature_importances = rf.feature_importances_
    
    # Select features with importance greater than mean importance
    mean_importance = np.mean(feature_importances)
    selected_features = feature_importances > mean_importance
    return selected_features

def QSAR_with_SVR_RF(df_descriptors, df_pIC50, save_dir, df_et):
    
    X = df_descriptors.values
    y = df_pIC50.values
    X_external_test = df_et.values
    test = []
    for fold in range(5):
        train_index, test_index = read_fold_indices(fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        selected_features = feature_selection_by_random_forest(X_train, y_train)
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        X_et = X_external_test[:, selected_features]
        scaler = StandardScaler()
        X_train_selected = scaler.fit_transform(X_train_selected)
        X_test_selected = scaler.transform(X_test_selected)
        X_et = scaler.transform(X_et)

        svr_model = SVR(kernel='rbf')  # Using RBF kernel for SVR
        svr_model.fit(X_train_selected, y_train)
        y_pred = svr_model.predict(X_test_selected)
        test.append(svr_model.predict(X_et))
        mse = mean_squared_error(y_test, y_pred)
        print(f"Fold {fold+1}, Mean Squared Error: {mse:.4f}")

        y_train_pred = svr_model.predict(X_train_selected)
        with open(os.path.join(save_dir, f'fold-{fold}-test.txt'), 'w') as f:
            for actual, predicted in zip(y_test, y_pred):
                f.write(f'{actual} {predicted}\n')
        
        with open(os.path.join(save_dir, f'fold-{fold}-train.txt'), 'w') as f:
            for actual, predicted in zip(y_train, y_train_pred):
                f.write(f'{actual} {predicted}\n')                
    np.savetxt(root_dir + '/RFR_SVR/test.txt', np.array(test).mean(axis=0).reshape(-1, 1))   
                


if __name__ == "__main__":
    descriptors_save_path = root_dir + '/dataset/descriptors.csv'
    input_file_path = root_dir + '/dataset/clean_dpp4_ic50.csv'
    save_dir = root_dir + '/RFR_SVR/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_pIC50 = pd.read_csv(input_file_path, sep='\t')['Label']
    df_descriptors, df_et = multicollinearity_analysis(descriptors_save_path)

    QSAR_with_SVR_RF(df_descriptors, df_pIC50, save_dir, df_et)
