import torch
import config
import dataset
import EDSR
import EDSR_trainer

# Load Image data
train_dataset, val_dataset, train_dataloader, val_dataloader = dataset.load_superres_data(config.IMAGE_FOLDER,
                                                                                          batch_size=config.BATCH_SIZE,
                                                                                          validation_split=config.VALIDATION_SPLIT)

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

# Train the model
history = trainer.train(train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        num_epochs=config.NUM_EPOCHS)

# Save the model and the training history
model.save_model(config.SAVING_FOLDER)
model.save_history(history, config.SAVING_FOLDER)



