from orchestrator import Orchestrator
from envs import OADEnv
from config import Config
from logger import LoggerServer

import time
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def env_runner(name: str, cfg: Config, queue: mp.Queue, debug: bool = True):
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

    for epoch in range(cfg.epochs):
        # TRAINING
        orchestrator.set_mode(train=True)
        obs = env.reset()
        orchestrator.reset_memory()
        for episode in range(cfg.train_episodes):
            choices = orchestrator.act(obs)
            obs, rewards, _, _ = env.step(choices)
            orchestrator.rewarding(rewards)
        orchestrator.learn()
        orchestrator.logger.call('action_map', None)
        if debug:
            print(f'{name}: training {epoch + 1}/{cfg.epochs} done')

        # EVALUATION
        orchestrator.set_mode(train=False)
        obs = env.reset()
        orchestrator.reset_memory()
        with torch.no_grad():
            for episode in range(cfg.test_episodes):
                choices = orchestrator.act(obs)
                obs, rewards, _, _ = env.step(choices)
                orchestrator.rewarding(rewards)
        orchestrator.logger.call('action_map', None)
        if debug:
            print(f'{name}: evaluation {epoch + 1}/{cfg.epochs} done')

    orchestrator.logger.call('policy_via_time', None)
    orchestrator.logger.call('coop_bars', None)
    orchestrator.logger.param({'spent time': time.time() - start_time})


if __name__ == '__main__':
    config = Config.init()

    logger_server = LoggerServer()
    logger_server.start()

    import dataclasses
    with ProcessPoolExecutor(max_workers=config.cores) as executor:
        runners = []
        lrs = [0.01, 0.005, 0.001, 0.0001]

        for lr in lrs:
            _config = dataclasses.replace(config)
            _config.set('lr', lr)
            runners.append(executor.submit(env_runner, str(lr), _config, logger_server.queue))

        for counter, runner in enumerate(as_completed(runners)):
            try:
                result = runner.result()
            except Exception as ex:
                raise ex

            print(f'Games finished: {counter + 1}/{config.games}')

    logger_server.stop()
