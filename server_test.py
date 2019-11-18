import json
import requests
import numpy as np
X_new = [[1,4,5,7,8],[2,4,5,9,10]]
print(X_new)
input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": X_new,
})
SERVER_URL = 'http://144.202.100.179:8501/v1/models/my_cls_model:predict'
response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status() # raise an exception in case of error
response = response.json()
y_proba = np.array(response["predictions"])
print(y_proba)