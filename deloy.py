from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage

ws = Workspace.from_config()
model = Model(ws, 'pytorch_mnist')

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,
                                               memory_gb=1,
                                               tags={"data": "MNIST", "method": "pytorch"},
                                               description='Predict MNIST with pytorch')
# configure the image
image_config = ContainerImage.image_configuration(execution_script="score.py",
                                                  runtime="python",
                                                  conda_file="myenv.yml")

service = Webservice.deploy_from_model(workspace=ws,
                                       name='pytorch-mnist-svc',
                                       deployment_config=aciconfig,
                                       models=[model],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)

print(service.scoring_uri)
