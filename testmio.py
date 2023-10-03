from challenge.model import DelayModel
import pandas as pd
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import train_test_split
import numpy as np

FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

OPERA_VALUES = {
    "Aerolineas Argentinas": 0,
    # Agregar otros valores de OPERA según sea necesario
}

TIPOVUELO_VALUES = {
    "N": 5,
    # Agregar otros valores de TIPOVUELO según sea necesario
}

MES_VALUES = {
    3: 1,
    # Agregar otros valores de MES según sea necesario
}
model = DelayModel()
data_path = os.path.join(os.path.dirname(__file__), './data/data.csv')
data = pd.read_csv(filepath_or_buffer=data_path,low_memory=False)
features, target = model.preprocess(data=data, target_column="delay")
_, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)
flight=[
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
matrix = pd.DataFrame(flight)
# Crear columnas de OPERA
initial_features = {
    "OPERA_Latin American Wings": 1,
    "MES_7": 1,
    "MES_10": 1,
    "OPERA_Grupo LATAM": 1,
    "MES_12": 1,
    "TIPOVUELO_I": 1,
    "MES_4": 1,
    "MES_11": 1,
    "OPERA_Sky Airline": 1,
    "OPERA_Copa Air": 1
}
features_test = pd.DataFrame(initial_features,index=[0])

features_test['OPERA_Latin American Wings'] = 1 if matrix['OPERA'].values[0] == 'Latin American Wings' else 0
features_test['MES_7'] = 1 if matrix['MES'].values[0] == 7 else 0
features_test['MES_10'] = 1 if matrix['MES'].values[0] == 10 else 0
features_test['OPERA_Grupo LATAM'] = 1 if matrix['OPERA'].values[0] == 'Grupo LATAM' else 0
features_test['MES_12'] = 1 if matrix['MES'].values[0] == 12 else 0
features_test['TIPOVUELO_I'] = 1 if matrix['TIPOVUELO'].values[0] == 'I' else 0
features_test['MES_4'] = 1 if matrix['MES'].values[0] == 4 else 0
features_test['MES_11'] = 1 if matrix['MES'].values[0] == 11 else 0
features_test['OPERA_Sky Airline'] = 1 if matrix['OPERA'].values[0] == 'Sky Airline' else 0
features_test['OPERA_Copa Air'] = 1 if matrix['OPERA'].values[0] == 'Copa Air' else 0
print(features_test)
model.fit(features=features, target=target)
features_test = features_test.apply(pd.to_numeric, errors='ignore')

# features_test = model.preprocess(data=data)
predicted_target = model.predict(features=features_test)
print(predicted_target)
# report = classification_report(target_validation, predicted_target, output_dict=True)
# print(report)