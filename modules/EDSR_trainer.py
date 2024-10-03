import torch
import torch.nn.functional as F
from tqdm import tqdm


class EDSR_Trainer:

    def __init__(self, model, optimizer, lr_scheduler, device):
        """ EDSR model trainer

        Parameters
        ----------
        model : nn.Module
            EDSR model
        optimizer : torch.optim
            Optimizer for the model
        lr_scheduler : torch.optim.lr_scheduler
            Learning rate scheduler
        device : torch.device
            Device where the model is stored
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    
    def train(self,
              train_dataloader,
              val_dataloader,
              num_epochs):
        """ Train the EDSR model

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Dataloader for the training set
        val_dataloader : torch.utils.data.DataLoader
            Dataloader for the validation set
        num_epochs : int
            Number of epochs for training

        Returns
        -------
        history : dict
            Dictionary with the training history
        """
        
        history = {"train_loss": [],
                   "val_loss": []}
        
        for epoch in range(num_epochs):
            
            # Training mode
            self.model.train()

            # Train one epoch
            train_loss = self._train_one_epoch(train_dataloader,
                                               epoch)
            
            # Update training history
            history["train_loss"].append(train_loss)


            # Validation 
            if val_dataloader is not None:
                
                # Evaluation mode
                self.model.eval()

                # Validation step
                val_loss = self._validate(val_dataloader)

                # Update validation history
                history["val_loss"].append(val_loss)
        
            # Update learning rate
            self.lr_scheduler.step()

        return history
    


    def _train_one_epoch(self, train_dataloader, epoch):
        """ Train the model for one epoch

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Dataloader for the training set
        epoch : int
            Current epoch number

        Returns
        -------
        mean_loss : float
            Mean loss for the epoch
        """
        
        running_loss = 0.
        mean_loss = 0.

        with tqdm(train_dataloader, unit="batches") as tepoch:

            for batch_idx, (imgs_lr, imgs_hr) in enumerate(tepoch):

                tepoch.set_description(f"Epoch {epoch+1}")

                # Load images on gpu
                imgs_lr = imgs_lr.to(self.device)
                imgs_hr = imgs_hr.to(self.device)

                # Set the gradient to zero
                self.optimizer.zero_grad()

                # Forward step
                pred_hr = self.model(imgs_lr)

                # Loss and gradient evaluation
                loss = F.l1_loss(imgs_hr, pred_hr)

                loss.backward()

                # Parameters optimization
                self.optimizer.step()

                # update running losses and mean losses
                running_loss += loss.item()
                mean_loss = running_loss / (batch_idx + 1)

                tepoch.set_postfix(mae_loss="{:.6f}".format(mean_loss))

        return mean_loss
    


    def _validate(self, val_dataloader):
        """ Validate the model on the validation set    

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            Dataloader for the validation set

        Returns
        -------
        mean_loss : float
            Mean loss for the validation set
        """
        running_loss = 0.
        mean_loss = 0.

        with torch.no_grad():

            for batch_idx, (imgs_lr, imgs_hr) in enumerate(val_dataloader):

                # Load images on gpu
                imgs_lr = imgs_lr.to(self.device)
                imgs_hr = imgs_hr.to(self.device)

                # Forward step
                pred_hr = self.model(imgs_lr)

                # Loss and gradient evaluation
                loss = F.l1_loss(imgs_hr, pred_hr)

                # update running losses and mean losses
                running_loss += loss.item()
                mean_loss = running_loss / (batch_idx + 1)

                # Print validation loss
                print(f"Validation loss: {mean_loss:.6f}")

        return mean_loss
 

        
        