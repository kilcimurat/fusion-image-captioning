import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from downloadDatasets.prepare_mscoco_dataset import MSCOCODataset
from downloadDatasets.prepare_vizwiz_dataset import VizWizDataset


from OneFeature.data_utils import get_loader_and_vocab
# from nets import *
from OneFeature.gnmt import GNMT


from tqdm import tqdm
import json

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# datasets = [MSCOCODataset(), VizWizDataset()]
dt = MSCOCODataset()
#dt = VizWizDataset()

val_annotation_file = dt.val_captions
val_annotation_name = str(val_annotation_file.parts[-1][:-5])
coco_val = COCO(str(val_annotation_file))

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader, test_loader, vocab = get_loader_and_vocab(dt, feature_type="ObjectDetection")

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
VOCABULARY_SIZE = vocab.__len__()
features, tokens = next(iter(train_loader))  
BATCH_SIZE, FEATURE_SIZE = features.shape
BATCH_SIZE, CAPTION_LENGTH = tokens.shape
HIDDEN_SIZE = 256
EMBED_SIZE = HIDDEN_SIZE

def train_step(net, X, y, optimizer, criterion):
    batch_size, caption_length = y.shape
    batch_size, feature_size = X.shape
    # X = X.unsqueeze(0)
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    
    decoder_hidden = torch.zeros((1, batch_size, 2*HIDDEN_SIZE), device=DEVICE)
    hiddens = []
    carrys = []
    for i in range(4):
        hiddens.append(decoder_hidden)
        carrys.append(decoder_hidden)
    loss = 0.0
    optimizer.zero_grad()
    for i in range(caption_length-1):
        output, hiddens, carrys = net.forward(y[:, i], hiddens, carrys, X)
        loss += criterion(output, y[:, i+1])
    loss.backward()
    optimizer.step()
    return loss.item() / caption_length



def test_step(net, X, y, vocab):
    batch_size, feature_size = X.shape
    # X = X.unsqueeze(0)
    X = X.to(DEVICE)
    decoder_hidden = torch.zeros((1, batch_size, 2*HIDDEN_SIZE), device=DEVICE)
    hiddens = []
    carrys = []
    for i in range(4):
        hiddens.append(decoder_hidden)
        carrys.append(decoder_hidden)
    loss = 0.0
    stoi = vocab.get_stoi()
    
    start_token = stoi['boc']
    end_token = stoi['eoc']
    result = torch.zeros((batch_size, CAPTION_LENGTH), dtype=torch.long, device=DEVICE)
    result[:, 0] = start_token
    for i in range(CAPTION_LENGTH-1):
        output, hiddens, carrys = net.forward(result[:, i], hiddens, carrys, X)
        result[:, i+1] = torch.argmax(output, dim=1)
    
    captions = []
    itos = vocab.get_itos()
    for i in range(batch_size):
        caption = ""
        tokens = result[i, :]
        for token in tokens[1:]:
            if token == end_token:
                break
            if itos[token][0] == "'":
                caption = caption[:-1] + itos[token] + " "
            else:
                caption += itos[token] + " "
        captions.append(caption)
    return captions, y





def train_model(net, vocab, epochs, comment, learning_rate=0.01):
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(net.parameters())
    criterion = nn.NLLLoss()
    writer = SummaryWriter(comment=comment)
    features, tokens = next(iter(train_loader))
    # features = features.unsqueeze(0)
    features = features.to(DEVICE)
    tokens = tokens.to(DEVICE)
    decoder_hidden = torch.zeros((1, BATCH_SIZE, 2*features.shape[-1]), device=DEVICE)
    hiddens = []
    carrys = []
    for i in range(4):
        hiddens.append(decoder_hidden)
        carrys.append(decoder_hidden)
    writer.add_graph(net, (tokens[:, 0], hiddens, carrys, features))
    BEST_VAL_CIDER_SCORE = 0.0
    for epoch in tqdm(range(epochs)):

        net.train()
        running_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            # X: features, y: tokens
            loss = train_step(net, X, y, optimizer, criterion)
            running_loss += loss
        print(running_loss / (i + 1))
        net.eval()
        val_data = []
        for X, y in val_loader:
            # X: features, y: ids
            with torch.no_grad():
                captions, ids = test_step(net, X, y, vocab)
            # print(f"id-{ids[0]}: {captions[0]}")
            for caption, id in zip(captions, ids):
                val_data.append({
                    "image_id": id.item(),
                    "caption" : caption
                })
            
            # raise NotImplementedError

        json_file = f"results/{dt.name}_DATASET_objectDetection_{val_annotation_name}_result.json"
        with open(json_file, "w") as file:
            json.dump(val_data, file)

        coco_val_result = coco_val.loadRes(json_file)
        coco_val_eval = COCOEvalCap(coco_val, coco_val_result)
        coco_val_eval.params['image_id'] = coco_val_result.getImgIds()
        coco_val_eval.evaluate()

        # print output evaluation scores
        for metric, score in coco_val_eval.eval.items():
            print(f'VAL {metric}: {score:.3f}')
            writer.add_scalar(f'VAL {metric}', score, epoch)
            if metric == "CIDEr":
                if BEST_VAL_CIDER_SCORE < score:
                    BEST_VAL_CIDER_SCORE = score
                    print(f"Best cider score is: {BEST_VAL_CIDER_SCORE}")
                    with open(f"results/{dt.name}_DATASET_objectDetection_{dt.name}_best_val_result.json", 'w') as file:
                        json.dump(val_data, file)
                    if test_loader is None:
                        continue

                    test_data = []
                    for X, y in test_loader:
                        # X: features, y: ids
                        with torch.no_grad():
                            captions, ids = test_step(net, X, y, vocab)
                        # raise NotImplementedError
                        for caption, id in zip(captions, ids):
                            test_data.append({
                                "image_id": id.item(),
                                "caption" : caption
                            })

                    json_file = f"results/{dt.name}_DATASET_objectDetection_{dt.name}_best_test_result.json"
                    with open(json_file, "w") as file:
                        json.dump(test_data, file)

        print(f"Best VAL cider score is: {BEST_VAL_CIDER_SCORE}")
       
        
        


    writer.close()

# net = GNMT(FEATURE_SIZE, HIDDEN_SIZE, VOCABULARY_SIZE, EMBED_SIZE, DEVICE).to(DEVICE)
# net = Decoder(FEATURE_SIZE, VOCABULARY_SIZE, EMBED_SIZE, 1)
net = GNMT(FEATURE_SIZE, HIDDEN_SIZE, VOCABULARY_SIZE)
net = net.to(DEVICE)
LEARNING_RATE = 0.01
EPOCH = 200
COMMENT = f"_ObjectDetection_{EPOCH}_EPOCH_{dt.name}_DATASET_{SEED}_SEED_NUMBER_{LEARNING_RATE}_LEARNING_RATE"

train_model(net, vocab, epochs=EPOCH, comment=COMMENT, learning_rate=LEARNING_RATE)


