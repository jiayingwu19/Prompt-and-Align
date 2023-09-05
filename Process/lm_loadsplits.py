import pandas as pd



def get_splits_fewshot(obj, n_samples):
    print("Fewshot: ", n_samples, " training samples")

    df_train = pd.read_csv("data/" + obj + "_train_" + str(n_samples) + ".csv", header=None, names = list(range(2)))
    df_test = pd.read_csv("data/" + obj + "_test.csv", header=None, names = list(range(2)))
    x_train = df_train[df_train.columns[1]].tolist()
    y_train = df_train[df_train.columns[0]].tolist()
    x_test = df_test[df_test.columns[1]].tolist()
    y_test = df_test[df_test.columns[0]].tolist()
    return x_train, x_test, y_train, y_test

