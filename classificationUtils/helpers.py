import torch
import numpy as np
import random
import os
from classificationUtils.args import args

from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def splitData(df):
    hotel_image_count = df.groupby("hotel_id")["image_id"].count()
    validation_hotels =  hotel_image_count[hotel_image_count>1]
    validation_data = df[df["hotel_id"].isin(validation_hotels.index)]
    validation_df = validation_data.groupby("hotel_id").sample(1,random_state=args.seed)
    validation_df = validation_df.reset_index(drop=True)
    train_df = df[~df["image_id"].isin(validation_df["image_id"])]
    # train_df.shape[0]+validation_df.shape[0]
    # print(f"Train data sample: {train_df.shape[0]} \nValidation data sample: {validation_df.shape[0]}")
    return train_df, validation_df


def getImgsDirs(df_splitted):
    imgs_dir = []
    for ind,row in df_splitted.iterrows():
        image_id = row["image_id"]
        hotel_id = row["hotel_id"]
        path = row["path"]
        imgs_dir.append((path,hotel_id,image_id))
    return imgs_dir

def decode_one_hot(y_one_hot):
    y = np.argmax(y_one_hot.cpu().numpy(), axis=1)
    return y

def generateFeatures(dataloader,model,improveEmbedding = False,colorFeat= None):
    features_all= []
    target_all=[]

    model.eval()
    with torch.no_grad():
        bar_description = "Generating embedding..."
        if(improveEmbedding):
             bar_description = "Extracting & improving embedding with {colorFeat}..."

        dataloader = tqdm(dataloader,desc=bar_description)
        for batch_no,(x, y,img_ids) in enumerate(dataloader):
                x = x.to(args.DEVICE)
                y = y.to(args.DEVICE)
                x = model(x)
                if(colorFeat):
                    color_feats = model.extractColorFeatures(img_ids,colorFeat)
                    x = model.fuseFeatures(x,color_feats)
                    target_all.extend(y.cpu().numpy())
                    features_all.extend(x.detach().cpu().numpy())
                else:
                    target_all.extend(y.cpu().numpy())
                    features_all.extend(x.detach().cpu().numpy())
                break
    target_all = np.array(target_all).astype(np.float32)
    features_all = np.array(features_all).astype(np.float32)
    return features_all,target_all


def save_checkpoint(model, scheduler, optimizer, epoch, name, loss=None, score=None):
    checkpoint = {"epoch": epoch,
                  "model": model.state_dict(),
                  "scheduler": scheduler.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "loss": loss,
                  "score": score,
                  }

    torch.save(checkpoint, f"{args.OUTPUT_FOLDER}checkpoint-{name}.pt")


def load_checkpoint(model, scheduler, optimizer, name):
    checkpoint = torch.load(f"{args.OUTPUT_FOLDER}checkpoint-{name}.pt")

    model.load_state_dict(checkpoint["model"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, scheduler, optimizer, checkpoint["epoch"]


def test_classification(loader, model,colorFeat= None):
    targets_all = []
    outputs_all = []
    outputs = None
    
    model.eval()
    dataloader = tqdm(loader, desc="Validation with Classification >>>")
    with torch.no_grad():
        for batch_no,(x, y,img_ids) in enumerate(dataloader):
            x = x.to(args.DEVICE)
            y = decode_one_hot(y)
            x = model(x)
            #improve embedding
            if(colorFeat=="rgb_feats" or colorFeat=="rgb_feats_10" or colorFeat=="rgb_feats_15"):
                color_feats = model.extractColorFeatures(img_ids,colorFeat)
                x = model.fuseFeatures(x,color_feats)
                outputs = model.rgbClassifier(x)
            elif(colorFeat=="hsv_feats" or colorFeat=="hsv_feats_10"or colorFeat=="hsv_feats_15"):
                color_feats = model.extractColorFeatures(img_ids,colorFeat)
                x = model.fuseFeatures(x,color_feats)
                outputs = model.hsvClassifier(x)

            elif(colorFeat in ["hist_feats_rgb_4", "hist_feats_hsv_4", "hist_feats_rgb_8","hist_feats_hsv_8","hist_feats_rgb_16","hist_feats_hsv_16"]):
                color_feats = model.extractColorFeatures(img_ids,colorFeat)
                x = model.fuseFeatures(x,color_feats)
                outputs = model.histClassifier(x)
            #use only embedding
            else:
                outputs = model.classifier(x)
            targets_all.extend(y)
            outputs_all.extend(torch.sigmoid(outputs).detach().cpu().numpy()) 
    
    # repeat targets to N_MATCHES for easy calculation of MAP@5
    y = np.repeat([targets_all], repeats=args.N_MATCHES, axis=0).T
    # sort predictions in ascending order i.e least class to top class
    sorted_indices = np.array(np.argsort(np.array(outputs_all),axis=1))
    # flip to sort in descending order and get top 5 classes i.e top class to least class 
    preds = np.flip(sorted_indices,1)[:,:args.N_MATCHES]
    preds = np.argsort(-np.array(outputs_all), axis=1)[:, :args.N_MATCHES]
    # check if any of top 5 predictions are correct and calculate mean accuracy
    acc_top_5 = (preds == y).any(axis=1).mean()
    # calculate prediction accuracy
    acc_top_1 = np.mean(targets_all == np.argmax(outputs_all, axis=1))

    print(f"Classification accuracy: {acc_top_1:0.4f}, MAP@5: {acc_top_5:0.4f}")
    return acc_top_1, acc_top_5

def trainEpoch(dataloader,model,criterion, optimizer, scheduler, epoch,classifier_to_use=None,color_feat_to_extract=None):
    targets_all=[]
    predicts_all = []
    losses = []

    model.train()
    t = tqdm(dataloader)

    for batch_no,(x, y,img_ids) in enumerate(t):
        optimizer.zero_grad()
        x = x.to(args.DEVICE)
        y = y.to(args.DEVICE)
    
        x = model(x)
        if(classifier_to_use=="rgb"):
            color_feats = model.extractColorFeatures(img_ids,feat=color_feat_to_extract)
            x = model.fuseFeatures(x,color_feats)
            outputs = model.rgbClassifier(x)
            loss = criterion(outputs,y)

            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            losses.append(loss.item())
            targets_all.extend(np.argmax(y.cpu().numpy(), axis=1))
            predicts_all.extend(torch.sigmoid(outputs).detach().cpu().numpy())

        elif(classifier_to_use=="hsv"):
            color_feats = model.extractColorFeatures(img_ids,feat=color_feat_to_extract)
            x = model.fuseFeatures(x,color_feats)
            outputs = model.hsvClassifier(x)
            loss = criterion(outputs,y)

            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            losses.append(loss.item())
            targets_all.extend(np.argmax(y.cpu().numpy(), axis=1))
            predicts_all.extend(torch.sigmoid(outputs).detach().cpu().numpy())

        elif(classifier_to_use=="hist"):
            color_feats = model.extractColorFeatures(img_ids,feat=color_feat_to_extract)
            x = model.fuseFeatures(x,color_feats)
            outputs = model.histClassifier(x)
            loss = criterion(outputs,y)

            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            losses.append(loss.item())
            targets_all.extend(np.argmax(y.cpu().numpy(), axis=1))
            predicts_all.extend(torch.sigmoid(outputs).detach().cpu().numpy())

        else:
            # classifier_to_use=="embedding"
            outputs = model.classifyWithEmbedding(x)
            loss = criterion(outputs,y)

            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            losses.append(loss.item())
            targets_all.extend(np.argmax(y.cpu().numpy(), axis=1))
            predicts_all.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        


        score = np.mean(targets_all == np.argmax(predicts_all, axis=1))
        desc = f"Training epoch {epoch}/{20} - batch loss:{loss:0.4f}, accuracy: {score:0.4f}"
        t.set_description(desc)
        
    return np.mean(losses), score