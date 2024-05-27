from pathlib import Path
import torch

class args:
    epoch = 20
    batch_size = 32
    lr=1e-3
    seed = 2024
    DEVICE = "mps" if torch.backends.mps.is_built() else "cpu"
    N_MATCHES = 5
    OUTPUT_FOLDER = str(Path().absolute().joinpath("models","classification","models"))+"/"
    ARTEFACT_FOLDER = str(Path().absolute().joinpath("artefacts","classification","logs"))+"/"
    IMG_SIZE = 224
    

# print(args.ARTEFACT_FOLDER)