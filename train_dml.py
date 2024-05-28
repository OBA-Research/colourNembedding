import warnings
warnings.filterwarnings("ignore")


from dmlUtils.args import args
from dmlUtils.helpers import getHooks,getTester,attachEndOfEpochHook, seed_everything,splitData, getImgsDirs
from dmlUtils.hotelsDataLoader import HOTELS, data_and_label_getter
from dmlUtils.models import getTrunk,getEmbedder,get_optimizers,get_schedulers,HotelTrainer


from pathlib import Path

import pandas as pd

from torch.utils.data import DataLoader

from pytorch_metric_learning import losses, miners, distances,reducers, samplers
import pytorch_metric_learning
import logging
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)
import faiss

seed_everything(args.seed)


tensorboardPath = Path().absolute().joinpath("artefacts/dml/tensorboard")
modelsPath = Path().absolute().joinpath("artefacts/dml/models")


df_pikcle_dir = Path().absolute().joinpath("dataset/randomHotels/randomHotelsFeats2.pkl")
df = pd.read_pickle(df_pikcle_dir)
args.df = df

# Split data into train and validation
train_df,validation_df = splitData(df)

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

print(f"train size = {len(train_dataset)}")
print(f"val size = {len(validation_dataset)}")

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


dataset_dict = {"train": train_dataset, "val": validation_dataset}

sampler = samplers.MPerClassSampler(
    train_df.hotel_id, m=4, length_before_new_iter=len(train_dataset)
)

###################Experiments########################################
Exps = [None]
Exps1 = ["hsv_feats","hsv_feats_11","hsv_feats_18","hsv_feats_28","hsv_feats_43","hsv_feats_64","hsv_feats_100"]
Exps2 = ["rgb_feats","rgb_feats_11","rgb_feats_18","rgb_feats_28","rgb_feats_43","rgb_feats_64","rgb_feats_100"]
Exp3 = ["hist_feats_rgb_5","hist_feats_hsv_5","hist_feats_rgb_11","hist_feats_hsv_11",
               "hist_feats_rgb_18","hist_feats_hsv_18","hist_feats_rgb_28","hist_feats_hsv_28",
               "hist_feats_rgb_43","hist_feats_hsv_43","hist_feats_rgb_64","hist_feats_hsv_64",
               "hist_feats_rgb_100","hist_feats_hsv_100"]
# Current focus
for focus in Exp3:
    args.COLOUR_FEAT = focus
    if(focus==None):
        logsPath = logsPath = Path().absolute().joinpath("artefacts/dml/logs/baseDml")
    else:
        logsPath = Path().absolute().joinpath(f"artefacts/dml/logs/{focus}")
    print(">>>>>>>>>>>>>>>>>>>> Experiment colour feature:",args.COLOUR_FEAT,"<<<<<<<<<<<<<<<<<<<<<<<")

    # Instantiate models
    trunk = getTrunk()
    embedder = getEmbedder(trunk_output_size=trunk.num_features,embedding_size=args.embedding_size)
    trunk_optimizer,embedder_optimizer = get_optimizers(trunk,embedder)
    trunk_schedule, embedder_schedule= get_schedulers(trunk_optimizer,embedder_optimizer,train_dataloader)


    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2)
    miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="hard")

    hooks = getHooks(logPath=logsPath,tensorboardPath=tensorboardPath)
    tester = getTester(hooks)
    end_of_epoch_hook = attachEndOfEpochHook(hooks,tester=tester,dataset_dict=dataset_dict,model_path=modelsPath)

    trainer = HotelTrainer(
        models={"trunk": trunk, "embedder": embedder},
        optimizers={"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer},
        batch_size=args.batch_size,
        loss_funcs={"metric_loss": loss_func},
        mining_funcs={"tuple_miner":miner},
        sampler = sampler,
        dataset=train_dataset,
        data_device=args.DEVICE,
        # data_and_label_getter = data_and_label_getter,
        dataloader_num_workers=args.N_WORKER,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
        lr_schedulers={
            'trunk_scheduler_by_iteration': trunk_schedule,
            'embedder_scheduler_by_iteration': embedder_schedule,
        },
        accumulation_steps=args.ACCUMULATION_STEPS,
        feats_df = args.df.copy(),
        improve_embeddings_with=args.COLOUR_FEAT
    )
    trainer.train(num_epochs=args.epoch)