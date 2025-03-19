import matplotlib
import pandas as pd
import numpy as np
import shap
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn.palettes as sns_palettes
from tqdm import tqdm
import seaborn as sns

# 1. 从CSV文件加载数据
data = pd.read_csv(r'C:\Users\CC\Desktop\SZ_data\data_sz_risk.csv', header=0)

datan = data
data_sample = data.sample(n=2000)
# RF
X = data_sample.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
x = datan.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
y = data_sample.iloc[:, 9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
ntree = 215
mtry = "log2"
RFModel = RandomForestRegressor(n_estimators=ntree, max_features=3, max_depth=5)  #

with tqdm(total=ntree, desc="Training Random Forest") as pbar:
    for i in range(ntree):
        pbar.update(1)
        RFModel.fit(X_train, y_train)
pbar.close()
print("Random Forest model trained")

# r2 MSE
y_pred = RFModel.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# RF important
feature_importances = RFModel.feature_importances_
factor_names = list(X.columns)
for i, importance in enumerate(feature_importances):
    print(f'Factor {factor_names[i]}: {importance}')

explainer = shap.TreeExplainer(RFModel)
shap_values_all = explainer(x)


# select colors
cmap = plt.get_cmap('YlGnBu')
colors = cmap(np.linspace(0.2, 0.8, 256))
summer_r_plus = matplotlib.colors.LinearSegmentedColormap.from_list('summer_r_plus', colors)
plt.figure()
shap.plots.beeswarm(shap_values_all, color=summer_r_plus)
plt.show()

# waterfall plot
shap.plots.waterfall(shap_values_all[2456143])

# summary plot
plt.figure()
shap.summary_plot(shap_values_all, X_train, plot_type="bar")
plt.savefig('shap_all_bar.png')
plt.show()

# dependence plot
shap.dependence_plot("LULC", shap_values_all, X, interaction_index=None, dot_size=2)   # change different factor

