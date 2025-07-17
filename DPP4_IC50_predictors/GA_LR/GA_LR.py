import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from genetic_selection import GeneticSelectionCV
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
    print(df_reduced.shape)
    print(df_test_reduced.shape)
    return df_reduced, df_test_reduced


def feature_selection_by_GA_sklearn(X_train, y_train, n_generations=20, population_size=30, mutation_prob=0.2, crossover_prob=0.5):
    """
    Feature selection using GeneticSelectionCV from sklearn-genetics.
    """
    estimator = LinearRegression()
    # GeneticSelectionCV is used to wrap the base estimator
    selector = GeneticSelectionCV(estimator,
                                  cv=5,  # 5-fold cross-validation
                                  verbose=1,
                                  scoring="neg_mean_squared_error",
                                  n_population=population_size,
                                  crossover_proba=crossover_prob,
                                  mutation_proba=mutation_prob,
                                  n_generations=n_generations,
                                  n_gen_no_change=10,  # Stop if no change for 10 generations
                                  caching=True,
                                  n_jobs=-1)  # Use all processors for parallelization

    selector = selector.fit(X_train, y_train)
    selected_features = selector.support_  # This will give a boolean array of selected features
    return selected_features

def QSAR_with_LR_GA(df_descriptors, df_pIC50, save_dir, df_test):
    
    X = df_descriptors.values
    y = df_pIC50.values
    X_external_test = df_test.values
    test = []
    for fold in range(5):
        train_index, test_index = read_fold_indices(fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        selected_features = feature_selection_by_GA_sklearn(X_train, y_train)
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
    
        test_features = X_external_test[:, selected_features]


        scaler = StandardScaler()
        X_train_selected = scaler.fit_transform(X_train_selected)
        X_test_selected = scaler.transform(X_test_selected)
        test_features = scaler.transform(test_features)

        lr_model = LinearRegression()
        lr_model.fit(X_train_selected, y_train)
        y_pred = lr_model.predict(X_test_selected)

        test.append(lr_model.predict(test_features))
        mse = mean_squared_error(y_test, y_pred)
        print(f"Fold {fold+1}, Mean Squared Error: {mse:.4f}")

        y_train_pred = lr_model.predict(X_train_selected)
        with open(os.path.join(save_dir, f'fold-{fold}-test.txt'), 'w') as f:
            for actual, predicted in zip(y_test, y_pred):
                f.write(f'{actual} {predicted}\n')
        
        with open(os.path.join(save_dir, f'fold-{fold}-train.txt'), 'w') as f:
            for actual, predicted in zip(y_train, y_train_pred):
                f.write(f'{actual} {predicted}\n')        
    np.savetxt(root_dir + '/GA_LR/test.txt', np.array(test).mean(axis=0).reshape(-1, 1))

if __name__ == "__main__":
    descriptors_save_path = root_dir + '/dataset/descriptors.csv'
    input_file_path = root_dir + '/dataset/clean_dpp4_ic50.csv'
    save_dir = root_dir + '/GA_LR/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_pIC50 = pd.read_csv(input_file_path, sep='\t')['Label']
    df_descriptors, df_test = multicollinearity_analysis(descriptors_save_path)

    QSAR_with_LR_GA(df_descriptors, df_pIC50, save_dir, df_test)



    