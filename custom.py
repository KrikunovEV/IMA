from logger import Metric, IArtifact
import matplotlib.pyplot as plt


class SumMetric(Metric):
    def __init__(self, key: str, log_on_train: bool = True, log_on_eval: bool = True, epoch_counter: bool = True):
        super(SumMetric, self).__init__(key, log_on_train, log_on_eval, epoch_counter)
        self.sum = 0

    # def set_mode(self, train: bool):
    #     self.sum = 0
    #     super(SumMetric, self).set_mode(train)

    def on_log(self, value):
        self.sum += value
        super(SumMetric, self).on_log(self.sum)


class ActionMap(IArtifact):

    def __init__(self, key: str, log_on_train: bool = True, log_on_eval: bool = True):
        super(ActionMap, self).__init__(key, log_on_train, log_on_eval)

    def on_log(self, data):
        fig = plt.figure(figsize=(16, 9))
        plt.close(fig)

    def on_all(self):
        pass
