from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from metric import Metric, BatchMetric
from mlflow_logger import LoggerServer, RunLogger


class CustomMetric(Metric):

    def __init__(self, key: str):
        super(CustomMetric, self).__init__(key)
        self.msgs = []

    def on_log(self, value):
        self.msgs.append(value)

    def on_all(self):
        pass

    def custom_func(self, data):
        print(self, max(self.msgs), data)


def test(queue: mp.Queue, i):
    _logger = RunLogger(queue)
    _logger.init(f'{i}', True, Metric('r'), CustomMetric('a'))
    _logger.param({'time': 0.5})
    [_logger.log({'r': i + j}) for j in range(10)]
    [_logger.log({'a': i * j}) for j in range(10)]
    _logger.call('custom_func', 'max value')
    _logger.deinit()


if __name__ == '__main__':
    logger_server = LoggerServer()
    logger_server.start()

    with ProcessPoolExecutor(max_workers=4) as executor:
        runners = [executor.submit(test, logger_server.queue, game) for game in range(1)]

        for counter, runner in enumerate(as_completed(runners)):
            try:
                result = runner.result()
            except Exception as ex:
                raise ex

    logger_server.stop()
