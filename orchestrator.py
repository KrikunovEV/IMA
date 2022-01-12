import numpy as np
import multiprocessing as mp

from logger import RunLogger
from custom import CoopsMetric, BatchSumAvgMetric, BatchAvgMetric, ActionMap, PolicyViaTime
from config import Config
from agents import get_agent_module
from elo_systems import MeanElo
from utils import indices_except, append_dict2dict


class Orchestrator:

    def __init__(self, o_space: int, a_space: int, cfg: Config, name: str, queue: mp.Queue):
        self.cfg = cfg
        self.mean_elo = MeanElo(cfg.players)
        self.train = None

        agent = get_agent_module(cfg.algorithm)
        self.agents = [agent(id=i, o_space=o_space, a_space=a_space, cfg=cfg) for i in range(cfg.players)]

        metrics = (
            ActionMap(cfg.actions_key, cfg.players, [agent.label for agent in self.agents], log_on_train=False),
            # PolicyViaTime(cfg.pvt_key, cfg.players, [agent.label for agent in self.agents], log_on_eval=False),
            CoopsMetric(cfg.actions_key, name, log_on_train=False)
        )
        for agent in self.agents:
            metrics += (BatchSumAvgMetric(f'{agent.label}_{cfg.reward_key}', 10),
                        BatchAvgMetric(f'{agent.label}_{cfg.elo_key}', 10, log_on_train=False, epoch_counter=True),
                        BatchAvgMetric(f'{agent.label}_{cfg.loss_key}', 10),
                        # BatchAvgMetric(f'{agent.label}_{cfg.eps_key}', 10),
                        )

        self.logger = RunLogger(queue, name, metrics)
        self.logger.param(cfg.as_dict())
        for agent in self.agents:
            agent.set_logger(self.logger)

    def __del__(self):
        self.logger.deinit()

    def set_mode(self, train: bool = True):
        self.train = train
        self.logger.set_mode(train)
        self.mean_elo.reset()
        for agent in self.agents:
            agent.set_mode(train)

    def act(self, obs):
        obs = self._preprocess(obs)
        logging_dict = {}
        actions = np.zeros((2, len(self.agents), len(self.agents)), dtype=np.int32)
        actions_key = self.cfg.actions_key
        for agent_id, agent in enumerate(self.agents):
            output = agent.act(obs) if self.train else agent.inference(obs)
            actions[:, agent_id, indices_except(agent_id, self.agents)] = output.pop(actions_key, None)
            if len(output) > 0:
                append_dict2dict(output, logging_dict)
        logging_dict[actions_key] = actions

        self.logger.log(logging_dict)
        if self.cfg.offend_policy_key in logging_dict and self.cfg.defend_policy_key in logging_dict:
            self.logger.log({self.cfg.pvt_key: (logging_dict[self.cfg.offend_policy_key],
                                                logging_dict[self.cfg.defend_policy_key])})

        return actions

    def rewarding(self, rewards, next_obs, last: bool):
        next_obs = self._preprocess(next_obs)
        for agent, reward in zip(self.agents, rewards):
            agent.rewarding(reward, next_obs, last)

        if self.train:
            for i, elo in enumerate(self.mean_elo.step(rewards)):
                self.logger.log({f'{self.agents[i].label}_{self.cfg.elo_key}': elo})

    def _preprocess(self, obs):
        obs = obs.flatten()
        return obs
