from ..trainer import BaseTrainer
import torch
from chronos import ChronosPipeline
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE
import numpy as np

class ChronosSmallTrainer(BaseTrainer):
    def __init__(self, device, prediction_length):
        self.pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.prediction_length = prediction_length

    def fit(self, train_dataloader):
        raise NotImplemented()
    
    def predict(self, test_dataloader):
        results = []
        for data in test_dataloader:
            context = torch.tensor(data[0]['encoder_target'])
            forecast = self.pipeline.predict(
                context,
                self.prediction_length,
                num_samples=10,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )
            result = torch.tensor(np.average(forecast, axis=1))
            results.append(result)
        return torch.concat(results)
    
    def eval(self, test_dataloader):
        output = self.predict(test_dataloader=test_dataloader)
        y = []
        for data in test_dataloader:
            y.append(data[0]['decoder_target'])

        y = torch.concat(y)

        return {
            "MAE": MAE()(y, output)[0],
            "RMSE": RMSE()(y, output)[0],
            "MAPE": MAPE()(y, output)[0],
            "SMAPE": SMAPE()(y, output)[0]
        }