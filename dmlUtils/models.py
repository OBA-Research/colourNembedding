import timm
import torch
from torch import nn
from dmlUtils.args import args

from pytorch_metric_learning import trainers
from pytorch_metric_learning.utils import common_functions as c_f
import numpy as np

import logging
logger = logging.getLogger(args.LOGGER_NAME)

class HotelTrainer(trainers.MetricLossOnly):
    def __init__(self, *args, accumulation_steps=10,feats_df = None, improve_embeddings_with=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps
        self.feats_df = feats_df
        self.colourKey = improve_embeddings_with
        self.img_ids = None

    def forward_and_backward(self):
        self.zero_losses()
        self.update_loss_weights()
        self.calculate_loss(self.get_batch())
        self.loss_tracker.update(self.loss_weights)
        self.backward()
        self.clip_gradients()
        if ((self.iteration + 1) % self.accumulation_steps == 0) or ((self.iteration + 1) == np.ceil(len(self.dataset) / self.batch_size)):
            self.step_optimizers()
            self.zero_grad()
        if(self.iteration + 1==1 and self.colourKey):
            logger.info(f"Each Batch Embeddings will be fused with {self.colourKey} colour features!")
        

    
    def get_batch(self):
        self.dataloader_iter, curr_batch = c_f.try_next_on_generator(
            self.dataloader_iter, self.dataloader
        )
        data, labels, self.img_ids  = self.data_and_label_getter(curr_batch)
        # print(self.img_ids)
        # data, labels  = self.data_and_label_getter(curr_batch)
        labels = c_f.process_label(
            labels, self.label_hierarchy_level, self.label_mapper
        )
        return self.maybe_do_batch_mining(data, labels)
           
    def calculate_loss(self, curr_batch):
        # print(curr_batch)
        data, labels = curr_batch
        # with torch.autocast(device_type=args.DEVICE):
        embeddings = self.compute_embeddings(data)
        # print(embeddings)
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        # print(indices_tuple[1])
        self.losses["metric_loss"] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple
        )
    def compute_embeddings(self, data):
        trunk_output = self.get_trunk_output(data)
        embeddings = self.get_final_embeddings(trunk_output)
        if(self.colourKey):
            # Extract & Fuse embeddings and color features
            color_feature = self._extractColorFeatures()
            embeddings = self._fuseFeatures(embeddings,color_feature)
        return embeddings
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

######################Some helping/abstracting functions################################

def getTrunk(backbone_name="efficientnet_b4"):
    trunk = timm.create_model(model_name = backbone_name, pretrained = True)
    trunk.classifier = nn.Identity()
    trunk  = trunk.to(args.DEVICE)
    return trunk

def getEmbedder(trunk_output_size,embedding_size):
    embedder = nn.Linear(trunk_output_size, embedding_size).to(args.DEVICE)
    return embedder

def get_optimizers(trunk,embedder):
    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=args.lr)
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=args.lr)
    # loss_optimizer = torch.optim.Adam(trunk.parameters(), lr=args.lr)
    return trunk_optimizer, embedder_optimizer

def get_schedulers(trunk_optimizer,embedder_optimizer,train_dataloader):
    trunk_schedule = torch.optim.lr_scheduler.OneCycleLR(
                        trunk_optimizer,
                        max_lr=args.lr,
                        epochs=args.epoch,
                        steps_per_epoch=len(train_dataloader),
                        div_factor=10,
                        final_div_factor=1,
                        pct_start=0.1,
                        anneal_strategy="cos",
                    )
    embedder_schedule = torch.optim.lr_scheduler.OneCycleLR(
                        embedder_optimizer,
                        max_lr=args.lr,
                        epochs=args.epoch,
                        steps_per_epoch=len(train_dataloader),
                        div_factor=10,
                        final_div_factor=1,
                        pct_start=0.1,
                        anneal_strategy="cos",
                    )
    return trunk_schedule, embedder_schedule
