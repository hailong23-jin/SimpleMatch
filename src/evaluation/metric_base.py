

class MetricBase:
    def __init__(self) -> None:
        pass

    def update_metrics(self, outputs, batch):
        raise NotImplementedError('Not implement!')
    
    def get_metrics(self):
        raise NotImplementedError('Not implement!')