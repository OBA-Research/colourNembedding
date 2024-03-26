import warnings
warnings.filterwarnings("ignore")

from dmlUtils.args import args
from dmlUtils.helpers import seed_everything,splitData, getImgsDirs, test_dml, trainEpoch, save_checkpoint
from dmlUtils.hotelsDataLoader import HOTELS
from dmlUtils.models import EmbeddingModel

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from pytorch_metric_learning import losses, miners, distances, testers,reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import faiss


seed_everything(args.seed)

df_pikcle_dir = Path().absolute().joinpath("dataset/randomHotels/randomHotelsFeats2.pkl")
df = pd.read_pickle(df_pikcle_dir)
df = df.astype({"hotel_id":"str"})

# Split data into train and validation
train_df,validation_df = splitData(df)
print(f"Train data sample: {train_df.shape[0]} \nValidation data sample: {validation_df.shape[0]}")
# print(validation_df.head())

# extract training and validation paths
train_imgs_dir = getImgsDirs(train_df)
validation_imgs_dir = getImgsDirs(validation_df)
# print(len(train_imgs_dir))

unique_labels = df["hotel_id"].unique()
num_classes=len(unique_labels)
print(f"Working with {num_classes} random hotels")


print("train and validation data loading....")
train_dataset = HOTELS(train_imgs_dir,unique_labels)
validation_dataset = HOTELS(validation_imgs_dir,unique_labels)

train_dataloader = DataLoader(
    train_dataset,
    num_workers =0,
    batch_size = args.batch_size,
    shuffle = True
)
ref_dataloader = DataLoader(
    train_dataset,
    num_workers =0,
    batch_size = args.batch_size,
    shuffle = False
)
validation_dataloader = DataLoader(
    validation_dataset,
    num_workers =0,
    batch_size = args.batch_size,
    shuffle = False
)
print("Data Loaded successfully \n")

# for batch_no,(x, y,img_ids) in enumerate(validation_dataset):
#     print(img_ids)
#     break

# Instantiate model
model = EmbeddingModel(num_classes,df).to(args.DEVICE)


distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

accuracy_calculator = AccuracyCalculator(include=("precision_at_1","mean_average_precision"), k=5, device=torch.device("cpu"))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=args.lr,
                        epochs=args.epoch,
                        steps_per_epoch=len(train_dataloader),
                        div_factor=10,
                        final_div_factor=1,
                        pct_start=0.1,
                        anneal_strategy="cos",
                    )

acc_top_1, acc_top_5 = [],[]
train_loss = []
prev_valid_acc = 0

model_name = "dml_embedding-model-{args.IMG_SIZE}x{args.IMG_SIZE}"
counter = 0 

for epoch in tqdm(range(1, args.epoch+1),desc="Training >>>"):
    # training_loss = trainEpoch(train_dataloader,model,loss_func,miner,optimizer,scheduler,epoch,args.epoch)
    training_loss = trainEpoch(train_dataloader,model,loss_func,miner,optimizer,scheduler=None,epoch=epoch,epochs=args.epoch)
    train_loss.append(training_loss)
    print(f"train loss : {training_loss}")

    val_acc_top_1, val_acc_top_5 = test_dml(ref_dataloader,validation_dataloader, model,accuracy_calculator,improveEmbedding=False,colorFeat=None)
    acc_top_1.append(val_acc_top_1)
    acc_top_5.append(val_acc_top_5)

    if prev_valid_acc<val_acc_top_5:
        save_checkpoint(model, scheduler, optimizer, epoch, model_name, train_loss)
        print("model saved..!!")
        prev_valid_acc = val_acc_top_5
        counter = 0
    else:
        counter +=1
    if(counter==5):
        print(f"early stopping applied, training done @ epoch {epoch}")
        break
    print(f".............................{epoch} end............................\n")

result_df = pd.DataFrame({"acc_top_1":acc_top_1,"acc_top_5":acc_top_5,"train_loss":train_loss})
result_df.to_csv(args.ARTEFACT_FOLDER+"dml_metrics_df.csv",index=False)
print(f">>>>>>>>>>>>>>>>> Experiment is Done!!!!!!!")