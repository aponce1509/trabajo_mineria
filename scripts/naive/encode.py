import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.preprocessing import OrdinalEncoder

def data_read_train(
    features_keep=None,
    features_drop=None,
    y_data_style="h1n1"
    ):
    x_data = pd.read_csv("data/training_set_features.csv", dtype='category')
    features = x_data.columns.values
    if y_data_style == "ml":
        y_data = pd.read_csv(
            "data/training_set_labels_4.csv", dtype='category'
        )
    else:
        y_data = pd.read_csv("data/training_set_labels.csv", dtype='category')
    if features_keep == None and features_drop == None:
        pass
    elif features_keep == None and features_drop != None:
        x_data = x_data.drop(features_drop, axis=1)
    elif features_keep != None and features_drop == None:
        # features_keep = features in features_keep
        x_data = x_data[features_keep]
    elif features_keep != None and features_drop != None:
        features_keep = [
            i for i in features_keep if i not in features_drop
        ]
        x_data = x_data[features_keep]
    
    if y_data_style == "h1n1":
        y_data = y_data.iloc[:, 1]
    elif y_data_style == "seasonal":
        y_data = y_data.iloc[:, 2]
    return x_data, y_data

def data_read_test(features_keep=None, features_drop=None):
    x_data = pd.read_csv("data/test_set_features.csv", dtype='category')
    features = x_data.columns.values
    if features_keep == None and features_drop == None:
        pass
    elif features_keep == None and features_drop != None:
        x_data = x_data.drop(features_drop, axis=1)
    elif features_keep != None and features_drop == None:
        # features_keep = features in features_keep
        x_data = x_data[features_keep]
    elif features_keep != None and features_drop != None:
        features_keep = [
            i for i in features_keep if i not in features_drop
        ]
        x_data = x_data[features_keep]
    return x_data


def ordinal_encoder(data):
    features = data.columns.values
    is_repondent = False
    features_order = {
        "education": [1, 0, 3, 2],
        # el age group esta ordenado
        "income_poverty": [2, 0, 1]
    }
    for column in features_order:
        if column in features:
            orderer_cat = data[column].cat.categories[features_order[column]]
            data[column] = data[column].cat.set_categories(orderer_cat)
            data[column] = data[column].cat.rename_categories(
                np.arange(0, len(orderer_cat))
            )
    if "respondent_id" in features:
        respondent_id = data["respondent_id"]
        data = data.drop("respondent_id", axis=1)
        features = data.columns.values
        is_repondent = True
    encoder = OrdinalEncoder()
    data = pd.DataFrame(
        encoder.fit_transform(data),
        columns=features
    )
    if is_repondent:
        data = pd.concat([respondent_id, data], axis=1)
    return data

if __name__ == "__main__":
    # x_data = pd.read_csv(
    #     "data/training_set_features.csv",
    #     dtype='category'
    # )
    # x_data.drop(["respondent_id"], axis=1, inplace=True)
    # features = x_data.columns.values
    # y_data = pd.read_csv(
    #     "data/training_set_labels.csv",
    #     dtype='category'
    # )
    # # df = pd.concat([df, labels], axis=1)
    # # print(x_data.head())
    # print(x_data["education"])
    # x_data = ordinal_encoder(x_data)
    # print(x_data["education"])
    # keep_ = ["h1n1_concern", "age_group", "sex"]
    drop_ = ["sex"]
    print(data_read_train())