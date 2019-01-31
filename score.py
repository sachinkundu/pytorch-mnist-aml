import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from azureml.core.model import Model
from torchvision import transforms
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


transforms = transforms.Compose([transforms.ToTensor()])


def predict_image(net, image):
    image_tensor = transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_tensor = Variable(image_tensor)
    output = net(input_tensor)
    index = output.data.cpu().numpy().argmax()
    return index


def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('pytorch_mnist')
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    img = Image.fromarray(np.array(data, dtype=np.int32).reshape(28, 28))
    # make prediction
    y_hat = predict_image(model, img)
    return json.dumps(str(y_hat))
