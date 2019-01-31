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

uri = ""

ds = AzureMNIST('data/MNISTPrepareData/processed', train=False)
img, target = ds.__getitem__(np.random.randint(0, len(ds)))
pix = np.array(img).reshape(-1)

input_data = "{\"data\": " + str(list(pix)) + "}"

headers = {'Content-Type':'application/json'}
resp = requests.post(uri, input_data, headers=headers)

print("label:", target)
print("prediction:", resp.text)

