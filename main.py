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
    seed = np.abs(name.__hash__()) % 4294967296  # 2**32
    cfg.set('seed', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f'{name}: np seed = {np.random.get_state()[1][0]}, torch seed = {torch.get_rng_state()[0].item()}')

    start_time = time.time()

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
        if debug:
            print(f'{name}: evaluation {epoch + 1}/{cfg.epochs} done')

    print(f'{name}: finished training in {time.time() - start_time} seconds')


if __name__ == '__main__':
    config = Config.init()

    logger_server = LoggerServer()
    logger_server.start()

    with ProcessPoolExecutor(max_workers=config.cores) as executor:
        runners = [executor.submit(env_runner, str(game), config, logger_server.queue) for game in range(config.games)]

        for counter, runner in enumerate(as_completed(runners)):
            try:
                result = runner.result()
            except Exception as ex:
                raise ex

            print(f'Games finished: {counter + 1}/{config.games}')

    logger_server.stop()
