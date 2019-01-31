import time
from azureml.core import Workspace, Experiment
from aml_compute import get_or_create_compute
from prepare_mnist_dataset import MNISTPrepareData
from azureml.train.dnn import PyTorch

# ws = Workspace._get_or_create(name='',
#                               subscription_id='',
#                               resource_group='',
#                               create_resource_group=True,
#                               location='westeurope'
#                               )

ws = Workspace.from_config()

experiment_name = "pytorch-mnist"
exp = Experiment(workspace=ws, name=experiment_name)

compute = get_or_create_compute(ws)

prepared_data = MNISTPrepareData(root="./data")

ds = ws.get_default_datastore()
# ds.upload(src_dir=prepared_data.processed_folder, target_path='mnist_pytorch', overwrite=True, show_progress=False)

script_params = {
    '--data-folder': ds.as_mount()
}

script_folder = './scripts'

est = PyTorch(source_directory=script_folder,
              script_params=script_params,
              compute_target=compute,
              entry_script='train.py')

run = exp.submit(config=est)
status = run.get_status()

while status != "Completed":
    if status in ["Failed", "Canceled"]:
        print('Run failed or cancelled')
        break
    else:
        print('Still running, Sleeping for a min and checking again...')
        time.sleep(60)
        status = run.get_status()

print(run.get_metrics())
# register model
model = run.register_model(model_name='pytorch_mnist', model_path='outputs/mnist_cnn.pt')
print(model.name, model.id, model.version, sep='\t')
