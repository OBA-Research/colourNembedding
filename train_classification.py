from classificationUtils.args import args
from classificationUtils.helpers import seed_everything,splitData, getImgsDirs, trainEpoch, test_classification, save_checkpoint
from classificationUtils.hotelsDataLoader import HOTELS
from classificationUtils.models import EmbeddingModel

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm


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
validation_dataloader = DataLoader(
    validation_dataset,
    num_workers =0,
    batch_size = args.batch_size,
    shuffle = False
)
print("Data Loaded successfully \n")

# Instantiate model
model = EmbeddingModel(num_classes,df).to(args.DEVICE)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
train_loss, train_score = [],[]
prev_valid_acc = 0

model_name = "classification_with_embedding-model-{args.IMG_SIZE}x{args.IMG_SIZE}"
counter = 0 
for epoch in tqdm(range(1, args.epoch+1),desc="Training >>>"):
    training_loss, training_score = trainEpoch(train_dataloader,model, criterion, optimizer, 
                                            scheduler, epoch)
    train_loss.append(training_loss)
    train_score.append(training_score)
    print(f"train loss : {training_loss} | train_acc : {training_score}")
    val_acc_top_1, val_acc_top_5 = test_classification(validation_dataloader, model)
    acc_top_1.append(val_acc_top_1)
    acc_top_5.append(val_acc_top_5)
    if prev_valid_acc<val_acc_top_5:
        save_checkpoint(model, scheduler, optimizer, epoch, model_name, train_loss, train_score)
        print("model saved..!!")
        prev_valid_acc = val_acc_top_5
        counter = 0
    else:
        counter +=1
    if(counter==5):
        print(f"early stopping applied, training done @ epoch {epoch}")
        break
    print(f"\n.............................{epoch} end............................")

result_df = pd.DataFrame({"acc_top_1":acc_top_1,"acc_top_5":acc_top_5,"train_loss":train_loss,"train_score":train_score})
result_df.to_csv(args.ARTEFACT_FOLDER+"classification_metrics_df.csv",index=False)
print(f">>>>>>>>>>>>>>>>> Experiment is Done!!!!!!!")