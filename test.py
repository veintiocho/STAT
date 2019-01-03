import pandas as pd
import numpy as np

import tqdm

from sklearn import metrics


def get_pred(model, model_name, input_file):
    result_dict = []

    file = pd.read_csv(input_file)

    ids = file.iloc[:,0].values
    X = file.iloc[:,1:].values

    y_pred = model.predict(X)

    for i in tqdm.tqdm(range(len(ids))):
        result_dict.append({'id':ids[i],'categories':y_pred[i]})

    pred_df = pd.DataFrame(result_dict)
    sub_name = 'submissions/submission_' + model_name + '.csv'
    pred_df.to_csv(sub_name, index=False)

    print("submission generated!")
