import xgboost as xgb
if __name__ == '__main__':
    model = xgb.Booster()
    model.load_model("XGBoostClassificationModel")
    model.get_score()


import matplotlib.pyplot as plt

plt.hist(a,b)