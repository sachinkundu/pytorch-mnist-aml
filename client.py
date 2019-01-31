import requests
import numpy as np
from scripts.AzureMNIST import AzureMNIST

# from azureml.core import Workspace
# from azureml.core.webservice import Webservice
#
# ws = Workspace.from_config()
# service = Webservice(workspace=ws, name='pytorch-mnist-svc')
# uri = service.serialize()['scoringUri']
# print(uri)

uri = "http://52.137.31.36:80/score"

ds = AzureMNIST('data/MNISTPrepareData/processed')
img, target = ds.__getitem__(100)
pix = np.array(img).reshape(-1)

input_data = "{\"data\": " + str(list(pix)) + "}"

headers = {'Content-Type':'application/json'}
resp = requests.post(uri, input_data, headers=headers)

print("label:", target)
print("prediction:", resp.text)

