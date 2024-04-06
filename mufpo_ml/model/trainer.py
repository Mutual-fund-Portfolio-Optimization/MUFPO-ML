class BaseTrainer:
    def __init__(self):
        pass

    def fit(self):
        raise NotImplemented()
    
    def predict(self):
        raise NotImplemented()
    
    def eval(self):
        raise NotImplemented()