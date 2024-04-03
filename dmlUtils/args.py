from pathlib import Path
import torch

class args:
    epoch = 40
    batch_size = 32
    lr=1e-3
    seed = 2024
    DEVICE = "mps" if torch.backends.mps.is_built() else "cpu"
    N_MATCHES = 5
    OUTPUT_FOLDER = str(Path().absolute().joinpath("models"))+"/"
    ARTEFACT_FOLDER = str(Path().absolute().joinpath("artefacts"))+"/"
    IMG_SIZE = 224
    N_WORKER = 0
    PATIENCE = 2
    embedding_size = 128
    ACCUMULATION_STEPS = 10
    N_CLASSES = 1000
    