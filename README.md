# Enhanced Deep Residual Networks (EDSR) for image Super Resolution

This repository contains the pytorch implementation of the paper "Enhanced Deep Residual Networks for Single Image Super-Resolution" (https://arxiv.org/abs/1707.02921).

## Repository Structure
The files are organized as follows:
- `dataset.py`: Contains the dataset class for loading the high resolution and low resolution images.
- `EDSR.py`: Contains the implementation of the EDSR model.
- `EDSR_train.py`: Contains the training object implementation for the EDSR model.
- `train.py`: Contains the training script for the EDSR model.
- `config.py`: Contains the configuration for the training script.


## How to use
Import the necessary dependencies:
```python
import torch
import config
import dataset
import EDSR
import EDSR_trainer
```

Load the dataset:
```python
train_dataset, val_dataset, train_dataloader, val_dataloader = dataset.load_superres_data(config.IMAGE_FOLDER,
                                                                                          batch_size=config.BATCH_SIZE,
                                                                                          validation_split=config.VALIDATION_SPLIT)
```

Create the model, the optimizer, the learning rate scheduler and the trainer:
```python
# Create model and the optimizer
model = EDSR.EDSRModel(channels=config.CHANNELS,
                       num_resblocks=config.NUM_RES_BLOCK,
                       factor=config.FACTOR).to(config.DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Create a lerning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                          T_max=config.NUM_EPOCHS,
                                                          eta_min=1e-6,
                                                          verbose=True)

# Create the trainer
trainer = EDSR_trainer.EDSR_Trainer(model=model,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    device=config.DEVICE)
```

Train the model:
```python
history = trainer.train(train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        num_epochs=config.NUM_EPOCHS)
```

Save the model and the training history:
```python
model.save_model(config.SAVING_FOLDER)
model.save_history(history, config.SAVING_FOLDER)
```


## Dependencies
* python == 3.12
* pytorch == 2.4.1
* torchvision == 0.19.1 
* tqdm == 4.66.5
