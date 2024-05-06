import torch
import pytorch_metric_learning.utils.logging_presets as LP
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils import common_functions as c_f

import numpy as np
import random
import os
from dmlUtils.args import args
from dmlUtils.hotelsDataLoader import data_and_label_getter

import logging
logger = logging.getLogger(args.LOGGER_NAME)

import tqdm

class NewCalculator(AccuracyCalculator):
    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, 
                                                  query_labels[:, None], 5,
                                                  self.avg_of_avgs,self.return_per_class,
                                                  self.label_comparison_fn)
    def calculate_precision_at_4(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, 
                                                  query_labels[:, None], 4,
                                                  self.avg_of_avgs,self.return_per_class,
                                                  self.label_comparison_fn)
    def calculate_precision_at_3(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, 
                                                  query_labels[:, None], 3,
                                                  self.avg_of_avgs,self.return_per_class,
                                                  self.label_comparison_fn)
    def calculate_precision_at_2(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, 
                                                  query_labels[:, None], 2,
                                                  self.avg_of_avgs,self.return_per_class,
                                                  self.label_comparison_fn)

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_5","precision_at_4","precision_at_3","precision_at_2"] 
class HotelTester(testers.BaseTester):
    def __init__(self, *args,feats_df = None, improve_embeddings_with=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_ids = None
        self.feats_df = feats_df
        self.colourKey = improve_embeddings_with

    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        s, e = 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                img, label, self.img_ids = self.data_and_label_getter(data)
                # print(label)
                # print(self.img_ids)
                label = c_f.process_label(label, "all", self.label_mapper)
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, img)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(
                        len(dataloader.dataset),
                        label.size(1),
                        device=self.data_device,
                        dtype=label.dtype,
                    )
                    all_q = torch.zeros(
                        len(dataloader.dataset),
                        q.size(1),
                        device=self.data_device,
                        dtype=q.dtype,
                    )
                e = s + q.size(0)
                all_q[s:e] = q
                labels[s:e] = label
                s = e
        if(self.colourKey):
            logger.info(f"Embeddings fused with {self.colourKey} colour features!")
        return all_q, labels 
    
    def _extractColorFeatures(self,):
        """
        return color features
        """
        color_feature = []
        for img_id in self.img_ids:
            color_feats = self.feats_df[self.feats_df.image_id==img_id][self.colourKey].values[0]
            color_feature.append(color_feats)
        return color_feature

    def _fuseFeatures(self,features_embedding,color_feature):
        """
        return fused features i.e. embedding + color_features
        """
        fused_features = []
        for i,colorFeats in enumerate(color_feature):
            colorFeats =torch.tensor(colorFeats,dtype=torch.float).to(args.DEVICE)
            embedding = features_embedding[i]
            features = torch.cat((embedding,colorFeats))
            fused_features.append(features)
        return torch.stack(fused_features)
    
    def get_embeddings_for_eval(self, trunk_model, embedder_model, input_imgs):
        input_imgs = c_f.to_device(
            input_imgs, device=self.data_device, dtype=self.dtype
        )
        trunk_output = trunk_model(input_imgs)
        if self.use_trunk_output:
            return trunk_output
        embeddings = embedder_model(trunk_output)
        if(self.colourKey):
            # Extract & Fuse embeddings and color features
            color_feature = self._extractColorFeatures()
            embeddings = self._fuseFeatures(embeddings,color_feature)
        return embeddings

    ##############GlobalEmbeddingSpaceTester stuff start ###########################
    def do_knn_and_accuracies(
    self, accuracies, embeddings_and_labels, query_split_name, reference_split_names):
        (
            query_embeddings,
            query_labels,
            reference_embeddings,
            reference_labels,
        ) = self.set_reference_and_query(
            embeddings_and_labels, query_split_name, reference_split_names
        )
        self.label_levels = self.label_levels_to_evaluate(query_labels)

        for L in self.label_levels:
            curr_query_labels = query_labels[:, L]
            curr_reference_labels = reference_labels[:, L]
            a = self.accuracy_calculator.get_accuracy(
                query_embeddings,
                curr_query_labels,
                reference_embeddings,
                curr_reference_labels,
                self.ref_includes_query(query_split_name, reference_split_names),
            )
            for metric, v in a.items():
                keyname = self.accuracies_keyname(metric, label_hierarchy_level=L)
                accuracies[keyname] = v
        if len(self.label_levels) > 1:
            self.calculate_average_accuracies(
                accuracies,
                self.accuracy_calculator.get_curr_metrics(),
                self.label_levels,
            )
        ##############GlobalEmbeddingSpaceTester stuff end###########################



######################Some helping/abstracting functions################################
            
def getHooks(logPath,tensorboardPath):
    record_keeper, _, _ = LP.get_record_keeper(logPath,tensorboard_folder=tensorboardPath)
    hooks = LP.get_hook_container(record_keeper, primary_metric='mean_average_precision')
    return hooks

def getTester(hooks):
    tester = HotelTester(
    feats_df=args.df.copy(),
    improve_embeddings_with=args.COLOUR_FEAT,
    end_of_testing_hook=hooks.end_of_testing_hook,
    accuracy_calculator=NewCalculator(
        # include=['precision_at_1','precision_at_2','precision_at_3','precision_at_4','precision_at_5','mean_average_precision'],
        include=['precision_at_1','mean_average_precision'],
        device=torch.device("cpu"),
        k=5),
    dataloader_num_workers=args.N_WORKER,
    data_device=args.DEVICE,
    batch_size=args.batch_size,
    # data_and_label_getter = data_and_label_getter
)
    return tester

def attachEndOfEpochHook(hooks,tester,dataset_dict,model_path):
    end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, 
    dataset_dict,
    model_path,
    test_interval=1, 
    patience=args.PATIENCE, 
    splits_to_eval = [('val', ['train'])]
)
    return end_of_epoch_hook


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

def save_checkpoint(model, scheduler, optimizer, epoch, name, loss=None):
    checkpoint = {"epoch": epoch,
                  "model": model.state_dict(),
                  "scheduler": scheduler.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "loss": loss,
                  }

    torch.save(checkpoint, f"{args.OUTPUT_FOLDER}checkpoint-{name}.pt")


def load_checkpoint(model, scheduler, optimizer, name):
    checkpoint = torch.load(f"{args.OUTPUT_FOLDER}checkpoint-{name}.pt")

    model.load_state_dict(checkpoint["model"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, scheduler, optimizer, checkpoint["epoch"]

###################################### From scratch start #######################################################
def generateFeatures(dataloader,split_set,model,improveEmbedding = False,colorFeat= None):
    features_all= []
    target_all=[]

    model.eval()
    with torch.no_grad():
        bar_description = f"Generating {split_set} split embedding..."
        if(improveEmbedding):
                bar_description = f"Extracting & improving {split_set} embedding with {colorFeat}..."

        dataloader = tqdm(dataloader,desc=bar_description)
        for batch_no,(x, y,img_ids) in enumerate(dataloader):
                x = x.to(args.DEVICE)
                y = y.to(args.DEVICE)
                x = model(x)
                if(colorFeat):
                    print("color feature extraction...")
                    color_feats = model.extractColorFeatures(img_ids,colorFeat)
                    x = model.fuseFeatures(x,color_feats)
                    target_all.extend(y.cpu().numpy())
                    features_all.extend(x.detach().cpu().numpy())
                else:
                    target_all.extend(y.cpu().numpy())
                    features_all.extend(x.detach().cpu().numpy())
                
    # target_all = np.array(target_all)
    # features_all = np.array(features_all).astype(np.float32)
    target_all = torch.tensor(target_all)
    features_all = torch.tensor(features_all,dtype=float)
    return features_all,target_all


def test_dml(train_loader,test_loader, model,accuracy_calculator,improveEmbedding=False,colorFeat= None):
    train_embeddings, train_labels = generateFeatures(train_loader,"Train", model,improveEmbedding,colorFeat)
    test_embeddings, test_labels = generateFeatures(test_loader,"Test", model,improveEmbedding,colorFeat)

    print("Computing accuracy...")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    acc_top_1 = accuracies["precision_at_1"]
    acc_top_5 = accuracies["mean_average_precision"]

    print(f"Test set accuracy ---- Precision@1 = {acc_top_1}   MAP@5 = {acc_top_5}")
    
    return acc_top_1, acc_top_5


def trainEpoch(dataloader,model,loss_func,miner, optimizer, scheduler, epoch,epochs,color_feat_to_extract=None):
    losses = []

    model.train()
    t = tqdm(dataloader)

    for batch_no,(x, y,img_ids) in enumerate(t):
        x = x.to(args.DEVICE)
        y = y.to(args.DEVICE)
        optimizer.zero_grad()
        embeddings = model(x)

        #Improve embeddings
        if(color_feat_to_extract):
            color_feats = model.extractColorFeatures(img_ids,feat=color_feat_to_extract)
            embeddings = model.fuseFeatures(embeddings,color_feats)
        
        tupple_pairs = miner(embeddings, y)

        loss = loss_func(embeddings,y,tupple_pairs)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        losses.append(loss.item())
        if batch_no % 20 == 0:
            desc = f"Epoch {epoch}/{epochs} - batch {batch_no}: Loss:{loss:0.4f} Number of mined triplets = {miner.num_triplets}"
            t.set_description(desc)
        
    return np.mean(losses)
###################################### From scratch end #######################################################