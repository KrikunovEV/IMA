from orchestrator import Orchestrator
from envs import OADEnv
from config import Config
from logger import LoggerServer, RunLogger
from custom import AvgCoopsMetric

import time
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def env_runner(name: str, cfg: Config, queue: mp.Queue, _to_return: dict, debug: bool = True):
    start_time = time.time()

    seed = np.abs(name.__hash__()) % 4294967296  # 2**32
    cfg.set('seed', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if debug:
        print(f'{name}: np seed = {np.random.get_state()[1][0]}, torch seed = {torch.get_rng_state()[0].item()}')

    env = OADEnv(players=cfg.players, debug=False)
    orchestrator = Orchestrator(o_space=np.prod(env.observation_space.n).item(),
                                a_space=env.action_space.shape[0],
                                cfg=cfg, name=name, queue=queue)

    choices_eval_to_return = []

    for epoch in range(cfg.epochs):

        # TRAINING
        orchestrator.set_mode(train=True)
        obs = env.reset()
        for episode in range(cfg.train_episodes):
            choices = orchestrator.act(obs)
            obs, rewards = env.step(choices)
            orchestrator.rewarding(rewards, obs, (episode + 1) == cfg.train_episodes)
        orchestrator.logger.all()
        if debug:
            print(f'{name}: training {epoch + 1}/{cfg.epochs} done')

        # EVALUATION
        orchestrator.set_mode(train=False)
        obs = env.reset()
        choices_eval_to_return.append([])
        with torch.no_grad():
            for episode in range(cfg.test_episodes):
                choices = orchestrator.inference(obs, episode)
                choices_eval_to_return[-1].append(choices)
                obs, rewards = env.step(choices)
                orchestrator.rewarding(rewards, obs, (episode + 1) == cfg.test_episodes)
        orchestrator.logger.call('action_map', None)
        orchestrator.logger.all()
        if debug:
            print(f'{name}: evaluation {epoch + 1}/{cfg.epochs} done')

    orchestrator.logger.call('policy_via_time', None)
    orchestrator.logger.call('coop_bars', None)
    orchestrator.logger.param({'spent time': time.time() - start_time})
    return {'to_return': _to_return, 'acts': choices_eval_to_return}


if __name__ == '__main__':
    configs = Config.init()

    logger_server = LoggerServer()
    logger_server.start()

    cores = next(iter(configs.values())).cores
    with ProcessPoolExecutor(max_workers=cores) as executor:
        runners = []

        for i, (name, config) in enumerate(configs.items()):
            run_logger = RunLogger(logger_server.queue, f'repeats={config.repeats} {name}',
                                   (AvgCoopsMetric('acts', config, log_on_train=False),), train=False)

            for repeat in range(config.repeats):
                _name = f'r{repeat} {name}' if config.repeats > 1 else name
                to_return = {'run_logger': run_logger, 'last': (repeat + 1) == config.repeats}
                runners.append(executor.submit(env_runner, _name, config, logger_server.queue, to_return))

        for counter, runner in enumerate(as_completed(runners)):
            try:
                result = runner.result()
                run_logger = result['to_return']['run_logger']
                run_logger.log({'acts': result['acts']})
                if result['to_return']['last']:
                    run_logger.call('avg_coop_bars', None)
                    run_logger.deinit()
            except Exception as ex:
                raise ex

            print(f'Games finished: {counter + 1}/{len(runners)}')

    logger_server.stop()
