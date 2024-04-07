from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, NormalDistributionLoss
import lightning as pl
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE
import torch
import pandas as pd
from datetime import datetime
from ..trainer import BaseTrainer

class TFTTrainer(BaseTrainer):
    def __init__(self, train_dataset, trainer, device='gpu'):
        self.device = device
        self.trainer = trainer
        self.tft = TemporalFusionTransformer.from_dataset(
            # dataset
            train_dataset,
            # architecture hyperparameters
            hidden_size=128,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            # loss metric to optimize
            loss=NormalDistributionLoss(),
            # logging frequency
            log_interval=2,
            # optimizer parameters
            learning_rate=0.01,
            reduce_on_plateau_patience=4
        )

    def fit(self, train_dataloader, validate_dataloader):
        self.trainer.fit(
            self.tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=validate_dataloader
        )
    
    def predict(self, dataloader):
        output = self.tft.predict(dataloader).cpu()
        return output

    def eval(self, test_dataloader):
        y = []
        for data in test_dataloader:
            y.append(data[0]['decoder_target'])
            
        y = torch.concat(y)

        output = self.predict(test_dataloader).cpu()
        return {
            "MAE": MAE()(y, output).item(),
            "RMSE": RMSE()(y, output).item(),
            "MAPE": MAPE()(y, output).item(),
            "SMAPE": SMAPE()(y, output).item()
        }