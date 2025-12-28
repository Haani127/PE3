import pandas as pd
import numpy as np
import os
import sys
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from ML_Modules import data_scale, split_data
warnings.filterwarnings('ignore')
def main():
    file_name = input()
    try:
        file_path = os.path.join(sys.path[0], file_name)
        if not os.path.exists(file_path):
            print("Error: File not found.")
            return
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: {e}")
        return
    y = df['average_monthly_hours']
    X = df.drop(columns=['Department', 'salary', 'average_monthly_hours'])
    X_scaled = data_scale(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.2)
    param_grid = {
        'max_depth': [3, 5, 7, 9, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt_regressor = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=dt_regressor, 
        param_grid=param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    cv_mse = -grid_search.best_score_
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    target_variance = np.var(y)
    print(f"Cross-validated MSE (after tuning): {cv_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Variance of target: {target_variance:.3f}")
if __name__ == "__main__":
    main()


#----------------------------ML MODULES: data_scale FUNCTION----------------------------


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def data_scale(X_DT):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_DT)
    return pd.DataFrame(scaled_data, columns=X_DT.columns)
def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)