import timm
import torch
from torch import nn
from classificationUtils.args import args

class EmbeddingModel(nn.Module):
    def __init__(self, num_classes,features_dataframe,rgb_size=None,hsv_size=None,hist_size=None,embedding_size=128,
                 backbone_name="efficientnet_b0"):
        super().__init__()
        
        self.df = features_dataframe 
        # self.rgb_size = rgb_size
        # self.hsv_size = hsv_size
        # self.hist_size = hist_size   

        self.num_classes = num_classes 
        self.backbone = timm.create_model(model_name = backbone_name,num_classes=num_classes, pretrained = True)
        in_features = self.backbone.get_classifier().in_features

        self.backbone.classifier = nn.Identity()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.classifier = nn.Linear(embedding_size,num_classes)
        if(rgb_size):
            self.rgbClassifier = nn.Linear(rgb_size+embedding_size,num_classes)
        elif(hsv_size):
            self.hsvClassifier = nn.Linear(hsv_size+embedding_size,num_classes)
        elif(hist_size):
            self.histClassifier = nn.Linear(hist_size+embedding_size,num_classes)

    def forward(self,x):
            """
            Return embeddings
            """
            x = self.backbone(x)
            x = x.view(x.size(0),-1)
            x = self.embedding(x)
            return x
    
    def extractColorFeatures(self,image_ids,feat="rgb_feats"):
        """
        return color features
        """
        color_feature = []
        for img_id in image_ids:
            color_feats = self.df[self.df.image_id==img_id][feat].values[0]
            color_feature.append(color_feats)
        return color_feature

    def fuseFeatures(self,features_embedding,features_color):
        """
        return fused features i.e. embedding + color_features
        """
        fused_features = []
        for i,colorFeats in enumerate(features_color):
            colorFeats =torch.tensor(colorFeats,dtype=torch.float).to(args.DEVICE)
            embedding = features_embedding[i]
            features = torch.cat((embedding,colorFeats))
            fused_features.append(features)
        return torch.stack(fused_features)

    def classifyWithEmbedding(self,x):
        """
        return hotel class using just embeddings
        """
        hotel_class = self.classifier(x)
        return hotel_class

    def classifyWithFusedFeatures(self,fused_features,classifer_to_use):
        """
        return hotel class using improved embeddings 
        """
        if classifer_to_use=="rgb":
            hotel_class = self.rgbClassifier(fused_features)
            return hotel_class
        elif classifer_to_use=="hsv":
            hotel_class = self.hsvClassifier(fused_features)
            return hotel_class
        else:
            hotel_class = self.histClassifier(fused_features)
            return hotel_class
