from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_RUN_NOTE


def init(worker, instance: dict, run_name: str, metrics: list):
    if instance in worker.instances:
        raise Exception(f'Logger warning: two .init() calls are occurred.')

    if len(metrics) == 0:
        raise Exception(f'Logger: .init() has to input at least one \'metric\'.')

    run_id = worker.logger.create_run(worker.exp_id, tags={MLFLOW_RUN_NAME: run_name}).info.run_id
    worker.logger.set_tag(run_id, MLFLOW_RUN_NOTE, f'run_id: {run_id}')
    for metric in metrics:
        metric.set_mlflow(worker.logger, (worker.global_run_id if metric.is_global else run_id))
    worker.instances[instance] = dict(run_id=run_id, metrics=metrics)


def log(worker, instance: dict, data: dict):
    for key, value in data.items():
        for metric in worker.instances[instance]['metrics']:
            if metric.is_able_to_log() and metric.key == key:
                metric.on_log(value)


def set_mode(worker, instance: dict, mode: bool):
    for metric in worker.instances[instance]['metrics']:
        metric.set_mode(mode)


def on_all(worker, instance: dict):
    for metric in worker.instances[instance]['metrics']:
        if hasattr(metric, 'on_all'):
            metric.on_all()


def call(worker, instance: dict, func_name: str, args):
    for metric in worker.instances[instance]['metrics']:
        if hasattr(metric, f'{func_name}'):
            getattr(metric, f'{func_name}')(args)


def param(worker, instance: dict, data: dict):
    for key, value in data.items():
        worker.logger.log_param(worker.instances[instance]['run_id'], key, value)


def artifact(worker, instance: dict, src_path: str, dest_dir: str):
    worker.logger.log_artifact(run_id=worker.instances[instance]['run_id'], local_path=src_path, artifact_path=dest_dir)


def artifacts(worker, instance: dict, src_path: str, dest_dir: str):
    worker.logger.log_artifacts(run_id=worker.instances[instance]['run_id'], local_dir=src_path, artifact_path=dest_dir)


def deinit(worker, instance: dict):
    if instance in worker.instances:
        worker.logger.set_terminated(worker.instances[instance]['run_id'], worker.terminated)
        worker.instances.pop(instance)


def stop(worker):
    worker.logger.set_terminated(worker.global_run_id, worker.terminated)
    return 'stop'
