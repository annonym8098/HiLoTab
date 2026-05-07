import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_fit_transform(X_train, X_test):
    X_train = pd.DataFrame(X_train).copy()
    X_test = pd.DataFrame(X_test).copy()

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(exclude=["int64", "float64"]).columns

    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

        X_train = pd.get_dummies(X_train, columns=cat_cols)
        X_test = pd.get_dummies(X_test, columns=cat_cols)

        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train.values, X_test.values