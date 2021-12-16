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
    print(f'{name}: np seed = {np.random.get_state()[1][0]}, torch seed = {torch.get_rng_state()[0].item()}')

    env = OADEnv(players=cfg.players, debug=False)
    logger = RunLogger(queue)
    orchestrator = Orchestrator(o_space=np.prod(env.observation_space.n).item(),
                                a_space=env.action_space.shape[0],
                                cfg=cfg, name=name, logger=logger)

    choices_eval_to_return = []

    for epoch in range(cfg.epochs):

        # TRAINING
        orchestrator.set_mode(train=True)
        obs = env.reset()
        for episode in range(cfg.train_episodes):
            choices = orchestrator.act(obs)
            obs, rewards = env.step(choices)
            orchestrator.rewarding(rewards, obs, (episode + 1) == cfg.train_episodes)
        logger.all()
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
        logger.call('action_map', None)
        logger.all()
        if debug:
            print(f'{name}: evaluation {epoch + 1}/{cfg.epochs} done')

    logger.call('policy_via_time', None)
    logger.call('coop_bars', None)
    logger.param({'spent time': time.time() - start_time})
    return {'to_return': _to_return, 'acts': choices_eval_to_return}


if __name__ == '__main__':
    configs = Config.init()

    logger_server = LoggerServer()
    logger_server.start()

    cores = next(iter(configs.values())).cores
    with ProcessPoolExecutor(max_workers=cores) as executor:
        runners = []

        for i, (name, config) in enumerate(configs.items()):

            if config.repeats > 1:
                run_logger = RunLogger(logger_server.queue)
                run_logger.init(f'repeats_{name}', False, AvgCoopsMetric('acts', config, log_on_train=False))

                for repeat in range(config.repeats):
                    _name = f'r{repeat} {name}'
                    to_return = {'run_logger': run_logger, 'last': (repeat + 1) == config.repeats}
                    runners.append(executor.submit(env_runner, _name, config, logger_server.queue, to_return))
            else:
                to_return = None
                runners.append(executor.submit(env_runner, name, config, logger_server.queue, to_return))

        for counter, runner in enumerate(as_completed(runners)):
            try:
                result = runner.result()
                to_return = result['to_return']
                if to_return is not None:
                    run_logger = to_return['run_logger']
                    run_logger.log({'acts': result['acts']})
                    if to_return['last']:
                        run_logger.call('avg_coop_bars', None)
                        run_logger.deinit()
            except Exception as ex:
                raise ex

            print(f'Games finished: {counter + 1}/{len(runners)}')

    logger_server.stop()
