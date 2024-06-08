
import torch
from torchvision import transforms


from pathlib import Path
from downloadDatasets.prepare_mscoco_dataset import MSCOCODataset
from downloadDatasets.prepare_vizwiz_dataset import VizWizDataset
from PIL import Image

from tqdm import tqdm
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
from torchvision.transforms import ToTensor


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
PARAMS = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 6}

weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
model = maskrcnn_resnet50_fpn(weights=weights)
model = model.to(device)
backbone = model.backbone

IMAGE_SIZE = 299

vizwiz_dt = VizWizDataset()
mscoco_dt = MSCOCODataset()


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
        self.output_folder = output_folder / "InstanceSegmentation"
        self.model = model
        self.model = model.eval()
        self.IMAGE_SIZE = IMAGE_SIZE 
        self.params = params
        self.ids = list(input_folder.glob('*.jpg'))
        self.set = FeatureExtractionDataset(self.ids, self.input_folder, self.IMAGE_SIZE)
        self.generator = torch.utils.data.DataLoader(self.set, **self.params)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        if not self.output_folder.exists():
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            print(f"path: {str(self.output_folder)} is created.")


    def run(self) -> None:
        print(f"Extracting: {str(self.input_folder)}")
        for local_batch, local_ids in tqdm(self.generator):
            local_batch = local_batch.to(device) 
            images = list(image for image in local_batch)
            with torch.no_grad():
                predictions = model.backbone(local_batch)
                predictions = self.avgpool(predictions['pool']).squeeze()
                for i, feature in enumerate(predictions):
                    file_name = Path(local_ids[i]).stem
                    torch.save(feature.cpu(), self.output_folder / f'{file_name}.pt')
        print(f"{str(self.input_folder)} extracted.")



input_folders = [vizwiz_dt.train_folder, vizwiz_dt.val_folder, vizwiz_dt.test_folder,mscoco_dt.train_folder,mscoco_dt.val_folder,mscoco_dt.test_folder]
output_folders = [vizwiz_dt.train_features_folder, vizwiz_dt.val_features_folder, vizwiz_dt.test_features_folder, mscoco_dt.train_features_folder,mscoco_dt.val_features_folder, mscoco_dt.test_features_folder]


for inp, out in zip(input_folders, output_folders):
    x = FeatureExtraction(input_folder=inp, output_folder=out, model=model, IMAGE_SIZE=IMAGE_SIZE, params=PARAMS)
    x.run()

