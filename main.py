from orchestrator import Orchestrator
from envs import OADEnv
from config import Config
from logger import LoggerServer
from custom import CoopsMetric

import time
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def env_runner(name: str, cfg: Config, queue: mp.Queue, debug: bool = False):
    start_time = time.time()

    seed = np.abs(name.__hash__()) % 4294967296  # 2**32
    cfg.set('seed', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f'{name}: np seed = {np.random.get_state()[1][0]}, torch seed = {torch.get_rng_state()[0].item()}')

    env = OADEnv(players=cfg.players, debug=False)
    orchestrator = Orchestrator(o_space=np.prod(env.observation_space.n), a_space=env.action_space.shape[0],
                                cfg=cfg, queue=queue, name=name)

    choices_eval_to_return = []

    for epoch in range(cfg.epochs):
        # TRAINING
        orchestrator.set_mode(train=True)
        obs = env.reset()
        orchestrator.reset_memory()
        for episode in range(cfg.train_episodes):
            orchestrator.negotiation()
            choices = orchestrator.act(obs)
            obs, rewards, _, _ = env.step(choices)
            orchestrator.rewarding(rewards, (episode + 1) == cfg.train_episodes)
        orchestrator.learn()
        orchestrator.logger.call('action_map', None)
        if debug:
            print(f'{name}: training {epoch + 1}/{cfg.epochs} done')

        # EVALUATION
        orchestrator.set_mode(train=False)
        obs = env.reset()
        orchestrator.reset_memory()
        choices_eval_to_return.append([])
        with torch.no_grad():
            for episode in range(cfg.test_episodes):
                orchestrator.negotiation()
                choices = orchestrator.act(obs)
                choices_eval_to_return[-1].append(choices)
                obs, rewards, _, _ = env.step(choices)
                orchestrator.rewarding(rewards, (episode + 1) == cfg.test_episodes)
        orchestrator.logger.call('action_map', None)
        if debug:
            print(f'{name}: evaluation {epoch + 1}/{cfg.epochs} done')

    orchestrator.logger.call('policy_via_time', None)
    orchestrator.logger.call('coop_bars', None)
    orchestrator.logger.param({'spent time': time.time() - start_time})
    return choices_eval_to_return


if __name__ == '__main__':
    config = Config.init()

    import dataclasses
    _configs = []
    _names = []
    repeats = config.games
    for lr in [0.0005]:
        for h_space in [128]:
            _config = dataclasses.replace(config)
            _config.set('lr', lr)
            _config.set('h_space', h_space)
            _configs.append(_config)
            _names.append(f'{lr}_{h_space}')

    logger_server = LoggerServer('32episodes, 500epochs, sum reward, no neg')
    logger_server.start()

    from custom import AvgCoopsMetric
    from logger import RunLogger
    run_logger = RunLogger(logger_server.queue)
    run_logger.init('avg', False, AvgCoopsMetric('acts', repeats, config.epochs))

    with ProcessPoolExecutor(max_workers=config.cores) as executor:
        runners = []
        for repeat in range(repeats):
            for _name, _config in zip(_names, _configs):
                _name = f'r{repeat}_{_name}'
                runners.append(executor.submit(env_runner, _name, _config, logger_server.queue))

        for counter, runner in enumerate(as_completed(runners)):
            try:
                result = runner.result()
                run_logger.log({'acts': result})
            except Exception as ex:
                raise ex

            print(f'Games finished: {counter + 1}/{len(runners)}')

    run_logger.call('avg_coop_bars', None)
    run_logger.deinit()
    logger_server.stop()
