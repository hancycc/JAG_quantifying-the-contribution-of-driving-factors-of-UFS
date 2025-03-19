import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import rasterio

def replace_invalid_values(array, invalid_value):
    array = array.astype(float)
    array[array <= invalid_value] = np.nan
    return array

def dataset_unit(year):
    AI_path = r"F:\250221\SZ_ok\AI.tif"
    Elevation_path = r'F:\250221\SZ_ok\DEM.tif'
    FVC_path = r'F:\250221\SZ_ok\FVC.tif'
    ISP_path = r'F:\250221\SZ_ok\ISP.tif'
    lithology_path = r'F:\250221\SZ_ok\lithology.tif'
    LULC_path = r'F:\250221\SZ_ok\LULC.tif'
    SHDI_path = r'F:\250221\SZ_ok\SHDI.tif'
    Slope_path = r'F:\250221\SZ_ok\Slope.tif'
    SWR_path = r'F:\250221\SZ_ok\SWR.tif'
    point_path = r'F:\250221\SZ_ok\zpoint.tif'

    with rasterio.open(AI_path) as src:
        AI = src.read(1)  # 读取第1个波段

    AI = replace_invalid_values(AI.flatten(), -340000000)
    Elevation = replace_invalid_values(rasterio.open(Elevation_path).read(1).flatten(), -3400000000)
    FVC = replace_invalid_values(rasterio.open(FVC_path).read(1).flatten(), -34000000)
    ISP = replace_invalid_values(rasterio.open(ISP_path).read(1).flatten(), -3400000000)
    lithology = replace_invalid_values(rasterio.open(lithology_path).read(1).flatten(), -34000000)
    LULC = replace_invalid_values(rasterio.open(LULC_path).read(1).flatten(), -34000000)
    SHDI = replace_invalid_values(rasterio.open(SHDI_path).read(1).flatten(), -34000000000)
    Slope = replace_invalid_values(rasterio.open(Slope_path).read(1).flatten(), -3400000000)
    SWR = replace_invalid_values(rasterio.open(SWR_path).read(1).flatten(), -3400000000)
    point = replace_invalid_values(rasterio.open(point_path).read(1).flatten(), -128)

    data = pd.DataFrame({
        'AI': AI,
        'Elevation': Elevation,
        'FVC': FVC,
        'ISP': ISP,
        'lithology': lithology,
        'LULC': LULC,
        'SHDI': SHDI,
        'Slope': Slope,
        'SWR': SWR,
        'Target': point,  # 将 point 转为一维数组
    })
    data = data.dropna()
    return data

def compare_models(data, n_iterations=1000):
    results = []

    for i in range(n_iterations):
        # 提取正样本和负样本
        positive_samples = data[data['Target'] == 1]
        negative_samples = data[data['Target'] == -1]
        # 随机选择与正样本数目相同的负样本
        if len(negative_samples) > len(positive_samples):
            negative_samples = negative_samples.sample(n=len(positive_samples))
        else:
            positive_samples = positive_samples.sample(n=len(negative_samples))
        # 将正负样本组合成一个新的数据集
        data_combined = pd.concat([positive_samples, negative_samples])
        # 随机打乱数据
        data_shuffled = shuffle(data_combined)
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(data_shuffled.iloc[:, :-1], data_shuffled['Target'],
                                                            test_size=0.3,random_state=42)

        # 定义模型名
        model_name = f'model_{i}'
        # 训练随机森林模型
        rf_model = RandomForestClassifier(n_estimators=215, max_features='log2', max_depth =12,min_samples_split=5)

        rf_model.fit(X_train, y_train)

        y_train_pred_rf = rf_model.predict(X_train)
        y_test_pred_rf = rf_model.predict(X_test)

        kappa_train_rf = cohen_kappa_score(y_train, y_train_pred_rf)
        auc_train_rf = roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])
        kappa_test_rf = cohen_kappa_score(y_test, y_test_pred_rf)
        auc_test_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
        print(i)

        results.append({
            'Iteration': i + 1,
            'Model_Name': model_name,
            'Kappa_train_rf': kappa_train_rf,
            'AUC_train_rf': auc_train_rf,
            'Kappa_test_rf': kappa_test_rf,
            'AUC_test_rf': auc_test_rf,
        })

    # 输出结果到CSV文件
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_RRU_results.csv', index=False)


if __name__ == "__main__":
    data = dataset_unit(2020)
    compare_models(data, n_iterations=1000)
