from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ListwiseTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
    ):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler(device=DEVICE)

    def __call__(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        epoch: int,
        num_epochs: int,
    ):
        kwargs = dict(
            dataloader=trn_loader,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        trn_loss = self._epoch_trn_step(**kwargs)

        kwargs = dict(
            dataloader=val_loader,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        val_loss = self._epoch_val_step(**kwargs)

        return trn_loss, val_loss

    def _epoch_trn_step(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        num_epochs: int,
    ):
        self.model.train()

        epoch_loss = 0.0

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs} TRN"
        )

        for user_idx, pos_idx, neg_idx in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                pos_idx=pos_idx.to(DEVICE), 
                neg_idx=neg_idx.to(DEVICE),
            )

            # forward pass
            with autocast(DEVICE.type):
                batch_loss = self._batch_step(**kwargs)

            # backward pass
            self._run_backprop(batch_loss)

            # accumulate loss
            epoch_loss += batch_loss.item()

        return epoch_loss / len(dataloader)

    @torch.no_grad()
    def _epoch_val_step(        
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        num_epochs: int,
    ):
        self.model.eval()

        epoch_loss = 0.0

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs} VAL"
        )

        for user_idx, pos_idx, neg_idx in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                pos_idx=pos_idx.to(DEVICE), 
                neg_idx=neg_idx.to(DEVICE),
            )

            # forward pass
            with autocast(DEVICE.type):
                batch_loss = self._batch_step(**kwargs)

            # accumulate loss
            epoch_loss += batch_loss.item()

        return epoch_loss / len(dataloader)

    def _batch_step(self, user_idx, pos_idx, neg_idx):
        pos_logit = self.model(user_idx, pos_idx)
        
        user_idx_exp = user_idx.unsqueeze(1).expand_as(neg_idx)
        neg_logit_flat = self.model(user_idx_exp.reshape(-1), neg_idx.reshape(-1))
        neg_logit = neg_logit_flat.view(*neg_idx.shape)
        
        loss = self.criterion(pos_logit, neg_logit)
        
        return loss

    def _run_backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()