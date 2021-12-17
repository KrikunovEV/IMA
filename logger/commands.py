

def init(worker, instance: str, run_name: str, metrics: tuple):
    if len(metrics) == 0:
        raise Exception(f'Logger: .init() has to input at least one \'metric\'.')

    names = []
    for metric in metrics:
        if metric._suffix_key in names:
            raise Exception(f'Logger: .init() got two metrics with the same suffix + key: {metric._suffix_key}.')
        else:
            names.append(metric._suffix_key)

    run_id = worker.create_run(run_name)
    worker.set_tag(run_id)
    for metric in metrics:
        metric.set_mlflow(worker.logger, (worker.global_run_id if metric.is_global else run_id))
    worker.instances[instance] = dict(run_id=run_id, metrics=metrics)


def log(worker, instance: str, data: dict):
    for key, value in data.items():
        for metric in worker.instances[instance]['metrics']:
            if metric.is_able_to_log() and metric.key == key:
                metric.on_log(value)


def set_mode(worker, instance: str, mode: bool):
    for metric in worker.instances[instance]['metrics']:
        metric.set_mode(mode)


def all(worker, instance: str):
    for metric in worker.instances[instance]['metrics']:
        metric.on_all()


def call(worker, instance: str, func_name: str, *args, **kwargs):
    for metric in worker.instances[instance]['metrics']:
        if hasattr(metric, f'{func_name}'):
            getattr(metric, f'{func_name}')(*args, **kwargs)


def param(worker, instance: str, data: dict):
    for key, value in data.items():
        worker.logger.log_param(worker.instances[instance]['run_id'], key, value)


def artifact(worker, instance: str, src_path: str, dest_dir: str):
    worker.logger.log_artifact(run_id=worker.instances[instance]['run_id'], local_path=src_path, artifact_path=dest_dir)


def artifacts(worker, instance: str, src_path: str, dest_dir: str):
    worker.logger.log_artifacts(run_id=worker.instances[instance]['run_id'], local_dir=src_path, artifact_path=dest_dir)


def deinit(worker, instance: str):
    worker.terminate_run(worker.instances[instance]['run_id'])
    worker.instances.pop(instance)


def stop(worker, instance: str):
    worker.terminate_run(worker.global_run_id)
    return 'stop'
