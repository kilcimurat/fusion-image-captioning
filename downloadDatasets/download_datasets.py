from downloadDatasets.prepare_mscoco_dataset import MSCOCODataset
from downloadDatasets.prepare_vizwiz_dataset import VizWizDataset

dt = MSCOCODataset()
dt.download_dataset()
#dt = VizWizDataset()
#dt.download_dataset()