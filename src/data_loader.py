import pandas as pd
import numpy as np

def preprocess_data(train_path, test_path, epsilon):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y = np.log1p(train_df['SalePrice'].values).reshape(-1, 1)

    test_id = test_df["Id"].values

    train_features = train_df.drop(columns=['Id', 'SalePrice'])
    test_features = test_df.drop(columns=['Id'])

    num_cols = train_features.select_dtypes(include=[np.number]).columns
    cat_cols = train_features.select_dtypes(exclude=[np.number]).columns

    train_median = train_features[num_cols].median()
    train_features[num_cols] = train_features[num_cols].fillna(train_median)
    test_features[num_cols] = test_features[num_cols].fillna(train_median)

    train_features[cat_cols] = train_features[cat_cols].fillna("None")
    test_features[cat_cols] = test_features[cat_cols].fillna("None")

    train_mean = train_features[num_cols].mean()
    train_std = train_features[num_cols].std()

    train_features[num_cols] = (train_features[num_cols] - train_mean) / (train_std + epsilon)
    test_features[num_cols] = (test_features[num_cols] - train_mean) / (train_std + epsilon)

    all_features = pd.concat([train_features, test_features], axis=0)
    all_features_encoded = pd.get_dummies(all_features)

    X_train = all_features_encoded.iloc[:len(train_df), :].values.astype(float)
    X_test = all_features_encoded.iloc[len(train_df):, :].values.astype(float)
    y = y.astype(float).reshape(-1, 1)
    
    return X_train, y, X_test, test_id