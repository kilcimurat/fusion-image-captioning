# Tries for dataloader and dataset on feature extraction.

import torch
from torchvision import transforms
from Inception import inception_v3


from pathlib import Path
from downloadDatasets.prepare_mscoco_dataset import MSCOCODataset
from downloadDatasets.prepare_vizwiz_dataset import VizWizDataset
from PIL import Image

from tqdm import tqdm



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
PARAMS = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 6}

model = inception_v3(pretrained=True)
model = model.to(device)

IMAGE_SIZE = 299

vizwiz_dt = VizWizDataset()


class FeatureExtractionDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ids, path, im_size):
        'Initialization'
        self.ids = ids
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        path = self.ids[index]
        # Load data and get label
        image = Image.open(path)
        image = image.convert('RGB')
        X = self.transform(image)
        y = str(path.parts[-1][:-4]) # get ID of the image from its path.

        return X, y

class FeatureExtraction():
    def __init__(self, input_folder:Path, output_folder:Path, model:torch.nn.Module, IMAGE_SIZE:int, params:dict) -> None:

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model = model.eval()
        self.IMAGE_SIZE = IMAGE_SIZE 
        self.params = params
        self.ids = list(input_folder.glob('*.jpg'))
        self.set = FeatureExtractionDataset(self.ids, self.input_folder, self.IMAGE_SIZE)
        self.generator = torch.utils.data.DataLoader(self.set, **self.params)
        if not self.output_folder.exists():
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            print(f"path: {str(self.output_folder)} is created.")


    def run(self) -> None:
        print(f"Extracting: {str(self.input_folder)}")
        for local_batch, local_ids in tqdm(self.generator):
            local_batch = local_batch.to(device) 
            with torch.no_grad():
                output = model(local_batch) # Shape (N, feature_size)
                for feature, id in zip(output, local_ids):
                    feature = feature.view(feature.shape[0], -1)
                    feature = feature.permute(1, 0)
                    torch.save(feature.cpu(), self.output_folder / f'{id}.pt')
        print(f"{str(self.input_folder)} extracted.")


input_folders = [vizwiz_dt.train_folder, vizwiz_dt.val_folder, vizwiz_dt.test_folder]
output_folders = [vizwiz_dt.train_features_folder, vizwiz_dt.val_features_folder, vizwiz_dt.test_features_folder]

for inp, out in zip(input_folders, output_folders):
    x = FeatureExtraction(input_folder=inp, output_folder=out, model=model, IMAGE_SIZE=IMAGE_SIZE, params=PARAMS)
    x.run()



