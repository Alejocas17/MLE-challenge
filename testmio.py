from challenge.model import DelayModel
import pandas as pd
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import train_test_split
model = DelayModel()
data_path = os.path.join(os.path.dirname(__file__), './data/data.csv')
data = pd.read_csv(filepath_or_buffer=data_path,low_memory=False)
features, target = model.preprocess(data=data, target_column="delay")
_, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

model.fit(features=features, target=target)
# features_test = model.preprocess(data=data)
predicted_target = model.predict(features=features_validation)

report = classification_report(target_validation, predicted_target, output_dict=True)
print(report)