import torch

IMAGE_FOLDER = 'Insert path to image folder here'
SAVING_FOLDER = 'Insert path to saving folder here'

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.05

CHANNELS = 64
NUM_RES_BLOCK = 16
FACTOR = 2

NUM_EPOCHS = 100
LEARNING_RATE = 1e-2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
