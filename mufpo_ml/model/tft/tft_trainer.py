from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, NormalDistributionLoss
import lightning as pl
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE
import torch
import pandas as pd
from datetime import datetime

class TFTTrainer:
    def __init__(self, train_dataset, device='gpu'):
        self.device = device

        self.early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=10, verbose=False, mode="min")
        self.trainer = pl.Trainer(
            max_epochs=20,
            accelerator=self.device ,
            devices=1,
            gradient_clip_val=0.1,
            limit_train_batches=30,
            limit_val_batches=3,
            callbacks=[self.early_stop_callback],
        )
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
        output = self.tft.predict(dataloader, return_index=True).cpu()
        temp = pd.DataFrame(output[0], columns=date_index, index=output[2].fund_name)
        new_df = temp.stack().reset_index(level=[0, 1])
        new_df = new_df.rename(columns={0: 'nav/unit_forecast','level_1': 'date'})
        return new_df

    def eval(self, test_dataloader):
        y = None
        for data in test_dataloader:
            if y == None:
                y = data[0]['decoder_target']
            else:
                y = torch.concat([data[0]['decoder_target'], y])

        output = self.tft.predict(test_dataloader).cpu()
        return {
            "MAE": MAE()(y, output),
            "RMSE": RMSE()(y, output),
            "MAPE": MAPE()(y, output),
            "SMAPE": SMAPE()(y, output)
        }