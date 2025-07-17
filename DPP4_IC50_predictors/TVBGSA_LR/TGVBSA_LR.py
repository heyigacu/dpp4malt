import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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


class TVBGSA:
    def __init__(self, fitness_function, n_agents, max_iter, n_features, mu_max=3, alpha=0.5):
        self.fitness_function = fitness_function  # Fitness function to evaluate agents
        self.n_agents = n_agents  # Number of agents (population size)
        self.max_iter = max_iter  # Maximum number of iterations
        self.n_features = n_features  # Number of features to select from
        self.mu_max = mu_max  # Maximum value for control parameter (mu)
        self.alpha = alpha  # Decay parameter for gravitational constant (G)
        self.G = 1  # Initial gravitational constant
        
        # Initialize agents with random binary positions (0 or 1)
        self.agents = np.random.randint(2, size=(n_agents, n_features))
        self.velocities = np.random.rand(n_agents, n_features)
        self.best_agent = None
        self.best_fitness = float('inf')
    
    def _update_gravitational_constant(self, iteration):
        # G decreases exponentially over time
        self.G = self.G * np.exp(-self.alpha * iteration / self.max_iter)
    
    def _calculate_fitness(self):
        # Calculate fitness for all agents
        fitnesses = np.array([self.fitness_function(agent) for agent in self.agents])
        return fitnesses
    
    def _update_positions(self, iteration):
        # Calculate the control parameter (mu) as a time-varying variable
        mu = self.mu_max / (iteration ** 2 if iteration > 0 else 1)
        
        # Calculate the probability function S(v) = tanh(v / mu)
        for i in range(self.n_agents):
            for j in range(self.n_features):
                prob = np.tanh(self.velocities[i, j] / mu)
                if np.random.rand() < prob:
                    # Switch between 0 and 1 with a probability
                    self.agents[i, j] = 1 - self.agents[i, j]
    
    def _update_velocities(self, fitnesses):
        # Sort agents by fitness
        sorted_indices = np.argsort(fitnesses)
        sorted_agents = self.agents[sorted_indices]
        sorted_fitnesses = fitnesses[sorted_indices]
        
        # Update best agent
        if sorted_fitnesses[0] < self.best_fitness:
            self.best_agent = sorted_agents[0]
            self.best_fitness = sorted_fitnesses[0]
        
        # Calculate masses and forces for all agents
        masses = sorted_fitnesses / np.sum(sorted_fitnesses)
        for i in range(self.n_agents):
            force = np.zeros(self.n_features)
            for j in range(self.n_agents):
                if i != j:
                    # Calculate Euclidean distance between agents i and j
                    distance = np.linalg.norm(sorted_agents[i] - sorted_agents[j])
                    # Calculate force exerted on agent i by agent j
                    force += self.G * masses[j] * (sorted_agents[j] - sorted_agents[i]) / (distance + 1e-8)
            # Update velocities based on calculated forces
            self.velocities[i] = np.random.rand(self.n_features) * self.velocities[i] + force
    
    def run(self):
        for iteration in range(self.max_iter):
            # Update gravitational constant
            self._update_gravitational_constant(iteration)
            
            # Calculate fitness for all agents
            fitnesses = self._calculate_fitness()
            
            # Update velocities based on fitness
            self._update_velocities(fitnesses)
            
            # Update positions of agents based on velocities and control parameter
            self._update_positions(iteration)
        
        return self.best_agent, self.best_fitness

def feature_selection_by_TVBGSA(X_train, y_train, n_generations=20, population_size=30, mu_max=3, alpha=0.5):
    n_features = X_train.shape[1]

    def fitness_function(agent):
        selected_features = np.where(agent == 1)[0]
        if len(selected_features) == 0:
            return float('inf')  # Penalize agents with no selected features
        X_selected = X_train[:, selected_features]
        model = LinearRegression()
        model.fit(X_selected, y_train)
        y_pred = model.predict(X_selected)
        mse = mean_squared_error(y_train, y_pred)
        return mse

    tvbgsa = TVBGSA(fitness_function, n_agents=population_size, max_iter=n_generations, n_features=n_features, mu_max=mu_max, alpha=alpha)
    best_agent, best_fitness = tvbgsa.run()
    selected_features = np.where(best_agent == 1)[0]
    return selected_features

def QSAR_with_LR_TVBGSA(df_descriptors, df_pIC50, save_dir, df_et):

    X = df_descriptors.values
    y = df_pIC50.values
    Xet = df_et.values
    test = []
    for fold in range(5):
        train_index, test_index = read_fold_indices(fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        selected_features = feature_selection_by_TVBGSA(X_train, y_train)
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        X_et = Xet[:, selected_features]
        scaler = StandardScaler()
        X_train_selected = scaler.fit_transform(X_train_selected)
        X_test_selected = scaler.transform(X_test_selected)
        X_et = scaler.transform(X_et)
        lr_model = LinearRegression()
        lr_model.fit(X_train_selected, y_train)
        y_pred = lr_model.predict(X_test_selected)
        test.append(lr_model.predict(X_et))
        mse = mean_squared_error(y_test, y_pred)
        print(f"Fold {fold+1}, Mean Squared Error: {mse:.4f}")

        y_train_pred = lr_model.predict(X_train_selected)
        with open(os.path.join(save_dir, f'fold-{fold}-test.txt'), 'w') as f:
            for actual, predicted in zip(y_test, y_pred):
                f.write(f'{actual} {predicted}\n')
        
        with open(os.path.join(save_dir, f'fold-{fold}-train.txt'), 'w') as f:
            for actual, predicted in zip(y_train, y_train_pred):
                f.write(f'{actual} {predicted}\n')  
    print(test)
    np.savetxt(root_dir + '/TVBGSA_LR/test.txt', np.array(test).mean(axis=0).reshape(-1, 1))         

if __name__ == "__main__":
    descriptors_save_path = root_dir + '/dataset/descriptors.csv'
    input_file_path = root_dir + '/dataset/clean_dpp4_ic50.csv'
    save_dir = root_dir + '/TVBGSA_LR/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_pIC50 = pd.read_csv(input_file_path, sep='\t')['Label']
    df_descriptors, df_et = multicollinearity_analysis(descriptors_save_path)

    QSAR_with_LR_TVBGSA(df_descriptors, df_pIC50, save_dir, df_et)
