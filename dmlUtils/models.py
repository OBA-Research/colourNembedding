import timm
import torch
from torch import nn
from classificationUtils.args import args

from pytorch_metric_learning import trainers
from pytorch_metric_learning.utils import common_functions as c_f
import numpy as np

class HotelTrainer(trainers.MetricLossOnly):
    def __init__(self, *args, accumulation_steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps

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

    
    def get_batch(self):
        self.dataloader_iter, curr_batch = c_f.try_next_on_generator(
            self.dataloader_iter, self.dataloader
        )
        # data, labels, img_ids  = self.data_and_label_getter(curr_batch)
        # print(type(curr_batch))
        data, labels  = self.data_and_label_getter(curr_batch)
        labels = c_f.process_label(
            labels, self.label_hierarchy_level, self.label_mapper
        )
        return self.maybe_do_batch_mining(data, labels)
           
    def calculate_loss(self, curr_batch):
        # data, labels,img_ids = curr_batch
        data, labels = curr_batch
        # with torch.autocast(device_type=args.DEVICE):
        embeddings = self.compute_embeddings(data)
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple
        )

def getTrunk(backbone_name="efficientnet_b0"):
    trunk = timm.create_model(model_name = backbone_name, pretrained = True)
    trunk.classifier = nn.Identity()
    trunk  = trunk.to(args.DEVICE)
    return trunk

def getEmbedder(trunk_output_size,embedding_size):
    embedder = nn.Linear(trunk_output_size, embedding_size).to(args.DEVICE)
    return embedder

def get_optimizers(trunk):
    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=args.lr)
    embedder_optimizer = torch.optim.Adam(trunk.parameters(), lr=args.lr)
    loss_optimizer = torch.optim.Adam(trunk.parameters(), lr=args.lr)

    return trunk_optimizer, embedder_optimizer,loss_optimizer

def get_schedulers(trunk_optimizer,embedder_optimizer,loss_optimizer,train_dataloader):
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
    loss_schedule = torch.optim.lr_scheduler.OneCycleLR(
                        loss_optimizer,
                        max_lr=args.lr,
                        epochs=args.epoch,
                        steps_per_epoch=len(train_dataloader),
                        div_factor=10,
                        final_div_factor=1,
                        pct_start=0.1,
                        anneal_strategy="cos",
                    )
    return trunk_schedule, embedder_schedule,loss_schedule

# class TrunkModel(nn.Module):
#     def __init__(self, num_classes,backbone_name="efficientnet_b0"):
#         super().__init__()
#         self.num_classes = num_classes 
#         self.backbone = timm.create_model(model_name = backbone_name,num_classes=num_classes, pretrained = True)
#         in_features = self.backbone.get_classifier().in_features
#         self.backbone.classifier = nn.Identity()

#     def forward(self,x):
#                 """
#                 Return embeddings
#                 """
#                 x = self.backbone(x)
#                 x = x.view(x.size(0),-1)
#                 # x = self.embedder(x)
#                 return x
    

        
# class TrunkModel(nn.Module):
#     def __init__(self, num_classes,features_dataframe,embedding_size=128,
#                  backbone_name="efficientnet_b0"):
#         super().__init__()
        
#         self.df = features_dataframe 


#         self.num_classes = num_classes 
#         self.backbone = timm.create_model(model_name = backbone_name,num_classes=num_classes, pretrained = True)
#         in_features = self.backbone.get_classifier().in_features

#         self.backbone.classifier = nn.Identity()
#         self.embedder = nn.Linear(in_features, embedding_size)

        
#     def forward(self,x):
#             """
#             Return embeddings
#             """
#             x = self.backbone(x)
#             x = x.view(x.size(0),-1)
#             # x = self.embedder(x)
#             return x
    
#     def extractColorFeatures(self,image_ids,feat="rgb_feats"):
#         """
#         return color features
#         """
#         color_feature = []
#         for img_id in image_ids:
#             color_feats = self.df[self.df.image_id==img_id][feat].values[0]
#             color_feature.append(color_feats)
#         return color_feature

#     def fuseFeatures(self,features_embedding,features_color):
#         """
#         return fused features i.e. embedding + color_features
#         """
#         fused_features = []
#         for i,colorFeats in enumerate(features_color):
#             colorFeats =torch.tensor(colorFeats,dtype=torch.float).to(args.DEVICE)
#             embedding = features_embedding[i]
#             features = torch.cat((embedding,colorFeats))
#             fused_features.append(features)
#         return torch.stack(fused_features)