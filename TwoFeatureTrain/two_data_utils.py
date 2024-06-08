# TODO tokenize captions first and then feed to the class Dataset.
# TODO document the processings and have comments in codes.
import torch

from text_processing import get_vocab, preprocess_txt

# Parameters
# Eva:
# PARAMS = {'batch_size': 64,
#           'shuffle': True,
#           'num_workers': 16}

# ozkan lab computer:
PARAMS = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 6}

class TestDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, paths, ids, feature_folder_one, feature_folder_two):
        'Initialization'
        self.paths = paths
        self.ids = ids
        self.feature_folder_one = feature_folder_one
        self.feature_folder_two = feature_folder_two

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.paths[index] + ".pt"
        feature_one= torch.load(self.feature_folder_one / name, map_location='cpu')
        feature_two = torch.load(self.feature_folder_two / name, map_location='cpu')
        feature = torch.cat((feature_two, feature_one), dim=-1)
        id = self.ids[index]

        return feature, id

class TrainDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ids, captions, vocab, tokenizer, preprocessor, feature_folder_one, feature_folder_two):
        'Initialization'
        self.ids = ids
        self.captions = captions
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.feature_folder_one = feature_folder_one
        self.feature_folder_two = feature_folder_two

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.ids[index] + ".pt"
        feature_one = torch.load(self.feature_folder_one / name, map_location='cpu')
        feature_two = torch.load(self.feature_folder_two / name, map_location='cpu')
        feature = torch.cat((feature_two, feature_one), dim=-1)
        tokens = self.captions[index]
        tokens = torch.LongTensor(tokens)

        return feature, tokens

def get_loader_and_vocab(dt, feature_type_one, feature_type_two):
    train_data, val_data, test_data = dt.load_data()
    processed_paths, processed_captions, vocab, tokenizer = get_vocab(train_data)
    train_feature_folder_one =dt.train_features_folder / feature_type_one
    train_feature_folder_two =dt.train_features_folder / feature_type_two
    train_dataset = TrainDataset(processed_paths, processed_captions, vocab, tokenizer, preprocess_txt, feature_folder_one=train_feature_folder_one, feature_folder_two=train_feature_folder_two)
    train_loader = torch.utils.data.DataLoader(train_dataset, **PARAMS)
    val_paths, val_ids = zip(*val_data)
    val_feature_folder_one =dt.val_features_folder / feature_type_one
    val_feature_folder_two =dt.val_features_folder / feature_type_two
    val_dataset = TestDataset(val_paths, val_ids, val_feature_folder_one, val_feature_folder_two)
    val_loader = torch.utils.data.DataLoader(val_dataset, **PARAMS)

    if test_data == None:
        test_loader = None
    else:
        test_paths, test_ids = zip(*test_data)
        test_feature_folder_one =dt.test_features_folder / feature_type_one
        test_feature_folder_two =dt.test_features_folder / feature_type_two
        test_dataset = TestDataset(test_paths, test_ids, test_feature_folder_one, test_feature_folder_two)
        test_loader = torch.utils.data.DataLoader(test_dataset, **PARAMS)
    return train_loader, val_loader, test_loader, vocab
