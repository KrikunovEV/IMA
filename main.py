from orchestrator import Orchestrator
from envs import get_environment
from config import Config
from logger import LoggerServer, RunLogger
from custom import AvgCoopsArtifact
from tqdm import tqdm

import time
import numpy as np
import torch
import multiprocessing as mp


def env_runner(name: str, cfg: Config, queue: mp.Queue, seed: int = None, debug: bool = True):
    start_time = time.time()

    cfg.set('seed', seed)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if debug:
            print(f'{name}: np seed = {np.random.get_state()[1][0]}, torch seed = {torch.get_rng_state()[0].item()}')

    env = get_environment(env_module=cfg.environment, players=cfg.players, debug=False)
    orchestrator = Orchestrator(o_space=env.observation_space,
                                a_space=env.action_space,
                                cfg=cfg, name=name, queue=queue)

    actions = []
    for epoch in range(cfg.epochs):

        # TRAINING
        obs = env.reset()
        orchestrator.reset(train=True)
        for e in tqdm(range(cfg.train_episodes), f'training {name}'):
            if cfg.enable_negotiation:
                orchestrator.negotiation()
            choices = orchestrator.act(obs)
            obs, rewards = env.step(choices)
            orchestrator.rewarding(rewards, obs, e == cfg.train_episodes - 1)
        orchestrator.logger.all()
        orchestrator.logger.call('all')
        orchestrator.logger.call('ema_plots')
        # if debug:
        #     print(f'{name}: training {epoch + 1}/{cfg.epochs} done')

        # TESTING
        obs = env.reset()
        orchestrator.reset(train=False)
        for e in tqdm(range(cfg.test_episodes), f'testing {name}'):
            if cfg.enable_negotiation:
                orchestrator.negotiation()
            choices = orchestrator.act(obs)
            actions.append(choices)
            obs, rewards = env.step(choices)
            orchestrator.rewarding(rewards, obs, e == cfg.test_episodes - 1)
        orchestrator.logger.all()
        orchestrator.logger.call('all')
        orchestrator.logger.call('action_map')
        # if debug:
        #     print(f'{name}: evaluation {epoch + 1}/{cfg.epochs} done')

    orchestrator.logger.param({'spent time': time.time() - start_time})
    return actions


if __name__ == '__main__':
    configs = Config.init()

    logger_server = LoggerServer(Config.experiment_name)
    logger_server.start()

    for i, (name, config) in enumerate(configs.items()):
        # logger = RunLogger(logger_server.queue, 'avg', (AvgCoopsArtifact(config.actions_key, config, name),))
        for repeat in range(config.repeats):
            run_name = f'{name} (r={repeat})' if config.repeats > 1 else name
            actions = env_runner(run_name, config, logger_server.queue)
            # logger.log({config.actions_key: actions})
        # logger.call('avg_coop_bars')
        # logger.deinit()
        print(f'Configs finished: {i + 1}/{len(configs)}')

    logger_server.stop()
