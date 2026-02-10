from IPython.display import clear_output
import torch
import torch.nn as nn


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Runner:
    def __init__(
        self, 
        model: nn.Module,
        trainer,
        monitor,
    ):
        self.model = model.to(DEVICE)
        self.trainer = trainer
        self.monitor = monitor

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        loo_loader: torch.utils.data.dataloader.DataLoader, 
        num_epochs: int, 
    ):
        kwargs = dict(
            trn_loader=trn_loader, 
            val_loader=val_loader, 
            loo_loader=loo_loader, 
            num_epochs=num_epochs, 
        )
        trn_loss_list, val_loss_list, loo_score_list = self._progressor(**kwargs)

        clear_output(wait=False)

        kwargs = dict(
            trn_loss_list=trn_loss_list, 
            val_loss_list=val_loss_list, 
            loo_score_list=loo_score_list,
        )
        return self._finalizer(**kwargs)

    def _progressor(self, trn_loader, val_loader, loo_loader, num_epochs):
        trn_loss_list = []
        val_loss_list = []
        loo_score_list = []

        for epoch in range(num_epochs):
            # trn, val
            kwargs = dict(
                trn_loader=trn_loader, 
                val_loader=val_loader, 
                epoch=epoch,
                num_epochs=num_epochs,
            )
            trn_loss, val_loss = self._run_trainer(**kwargs)

            # loo
            kwargs = dict(
                loo_loader=loo_loader,
                epoch=epoch,
                num_epochs=num_epochs,
            )
            loo_score = self._run_monitor(**kwargs)

            # accumulate
            trn_loss_list.append(trn_loss)
            val_loss_list.append(val_loss)
            loo_score_list.append(loo_score)

            # early stopping
            if self.monitor.should_stop==True:
                break

            # log reset
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        return trn_loss_list, val_loss_list, loo_score_list

    def _finalizer(self, trn_loss_list, val_loss_list, loo_score_list):
        if self.monitor.best_model_state is not None:
            self.model.load_state_dict(self.monitor.best_model_state)

        print(
            "LEAVE ONE OUT",
            f"\tBEST SCORE: {self.monitor.best_score:.4f}",
            f"\tBEST EPOCH: {self.monitor.best_epoch}",
            sep="\n",
        )

        return dict(
            trn=trn_loss_list,
            val=val_loss_list,
            loo=loo_score_list,
        )

    def _run_trainer(self, trn_loader, val_loader, epoch, num_epochs):
        kwargs = dict(
            trn_loader=trn_loader, 
            val_loader=val_loader, 
            epoch=epoch,
            num_epochs=num_epochs,
        )
        trn_loss, val_loss = self.trainer(**kwargs)

        print(
            f"TRN LOSS: {trn_loss:.4f}",
            f"VAL LOSS: {val_loss:.4f}",
            sep='\n'
        )

        return trn_loss, val_loss

    def _run_monitor(self, loo_loader, epoch, num_epochs):
        kwargs = dict(
            loo_loader=loo_loader, 
            epoch=epoch,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        loo_score = self.monitor(**kwargs)

        print(
            f"CURRENT SCORE: {loo_score:.4f}",
            f"BEST SCORE: {self.monitor.best_score:.4f}",
            f"BEST EPOCH: {self.monitor.best_epoch}",
            f"COUNTER: {self.monitor.counter}",
            sep='\t',
        )

        return loo_score