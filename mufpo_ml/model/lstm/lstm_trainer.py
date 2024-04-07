from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import RecurrentNetwork
import lightning as pl
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE
import torch
import pandas as pd
from datetime import datetime
from ..trainer import BaseTrainer

class LSTMTrainer(BaseTrainer):
    def __init__(self, train_dataset, trainer, date_index, device='gpu'):
        self.device = device
        self.trainer = trainer
        self.date_index = date_index
        self.rnn = RecurrentNetwork.from_dataset(
            train_dataset,
        )

    def fit(self, train_dataloader, validate_dataloader):
        self.trainer.fit(
            self.rnn,
            train_dataloaders=train_dataloader,
            val_dataloaders=validate_dataloader
        )
    
    def predict(self, dataloader):
        output = self.rnn.predict(dataloader).cpu()
        return output

    def eval(self, test_dataloader):
        y = []
        for data in test_dataloader:
            y.append(data[0]['decoder_target'])
            
        y = torch.concat(y)

        output = self.predict(test_dataloader)
        return {
            "MAE": MAE()(y, output),
            "RMSE": RMSE()(y, output),
            "MAPE": MAPE()(y, output),
            "SMAPE": SMAPE()(y, output)
        }