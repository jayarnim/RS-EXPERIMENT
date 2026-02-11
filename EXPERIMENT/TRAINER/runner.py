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
        dev_loader: torch.utils.data.dataloader.DataLoader, 
        cfg: dict, 
    ):
        NUM_EPOCHS = cfg["runner"]["num_epochs"]

        kwargs = dict(
            trn_loader=trn_loader, 
            val_loader=val_loader, 
            dev_loader=dev_loader, 
            num_epochs=NUM_EPOCHS, 
        )
        trn_log_list, val_log_list, dev_log_list = self._progressor(**kwargs)

        clear_output(wait=False)

        kwargs = dict(
            trn_log_list=trn_log_list, 
            val_log_list=val_log_list, 
            dev_log_list=dev_log_list,
        )
        return self._finalizer(**kwargs)

    def _progressor(self, trn_loader, val_loader, dev_loader, num_epochs):
        trn_log_list = []
        val_log_list = []
        dev_log_list = []

        for epoch in range(num_epochs):
            # trn, val
            kwargs = dict(
                trn_loader=trn_loader, 
                val_loader=val_loader, 
                epoch=epoch,
                num_epochs=num_epochs,
            )
            trn_loss, val_loss = self._run_trainer(**kwargs)

            # dev
            kwargs = dict(
                dev_loader=dev_loader,
                epoch=epoch,
                num_epochs=num_epochs,
            )
            dev_score = self._run_monitor(**kwargs)

            # accumulate
            trn_log_list.append(trn_loss)
            val_log_list.append(val_loss)
            dev_log_list.append(dev_score)

            # early stopping
            if self.monitor.should_stop==True:
                break

            # log reset
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        return trn_log_list, val_log_list, dev_log_list

    def _finalizer(self, trn_log_list, val_log_list, dev_log_list):
        if self.monitor.best_model_state is not None:
            self.model.load_state_dict(self.monitor.best_model_state)

        print(
            "DEVELOPMENT",
            f"\tBEST SCORE: {self.monitor.best_score:.4f}",
            f"\tBEST EPOCH: {self.monitor.best_epoch}",
            sep="\n",
        )

        return dict(
            trn=trn_log_list,
            val=val_log_list,
            dev=dev_log_list,
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

    def _run_monitor(self, dev_loader, epoch, num_epochs):
        kwargs = dict(
            dev_loader=dev_loader, 
            epoch=epoch,
            num_epochs=num_epochs,
        )
        dev_score = self.monitor(**kwargs)

        print(
            f"CURRENT METRIC: {dev_score:.4f}",
            f"BEST METRIC: {self.monitor.best_score:.4f}",
            f"COUNTER: {self.monitor.counter}",
            sep='\t',
        )

        return dev_score