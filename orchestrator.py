import numpy as np
import multiprocessing as mp

from logger import RunLogger
from custom import CoopsMetric, BatchSumAvgMetric, BatchAvgMetric, ActionMap, PolicyViaTime
from config import Config
from agents.q_learning import Agent
from elo_systems import MeanElo
from utils import indices_except, append_dict2dict


class Orchestrator:

    def __init__(self, o_space: int, a_space: int, cfg: Config, name: str, queue: mp.Queue):
        self.cfg = cfg
        self.agents = [Agent(id=i, o_space=o_space, a_space=a_space, cfg=cfg) for i in range(cfg.players)]
        self.mean_elo = MeanElo(cfg.players)

        metrics = (
            ActionMap('acts', cfg.players, [agent.label for agent in self.agents], log_on_train=False),
            PolicyViaTime('pvt', cfg.players, [agent.label for agent in self.agents], log_on_eval=False),
            CoopsMetric('acts', name, log_on_train=False, is_global=True)  # is_global=True to view in global run
        )
        for agent in self.agents:
            metrics += (BatchSumAvgMetric(f'{agent.label}_reward', 10, epoch_counter=False),
                        BatchAvgMetric(f'{agent.label}_elo', 10, log_on_train=False),
                        BatchAvgMetric(f'{agent.label}_loss', 10, epoch_counter=False),)

        self.logger = RunLogger(queue, name, metrics)
        self.logger.param(cfg.as_dict())
        for agent in self.agents:
            agent.set_logger(self.logger)

    def __del__(self):
        self.logger.deinit()

    def set_mode(self, train: bool = True):
        self.logger.set_mode(train)
        self.mean_elo.reset()
        for agent in self.agents:
            agent.set_mode(train)

    def act(self, obs):
        return self._make_act(obs, True)

    def inference(self, obs, episode):
        return self._make_act(obs, episode >= self.cfg.window)

    def rewarding(self, rewards, next_obs, last: bool):
        for i, elo in enumerate(self.mean_elo.step(rewards)):
            self.logger.log({f'{self.agents[i].label}_elo': elo[i]})

        next_obs = self._preprocess(next_obs)
        for agent, reward in zip(self.agents, rewards):
            agent.rewarding(reward, next_obs, last)

    def _make_act(self, obs, do_log):
        obs = self._preprocess(obs)
        logging_dict = {}
        actions = np.zeros((2, len(self.agents), len(self.agents)), dtype=np.int32)
        actions_key = self.cfg.actions_key
        for agent_id, agent in enumerate(self.agents):
            output = agent.act(obs) if agent.train else agent.inference(obs)
            actions[:, agent_id, indices_except(agent_id, self.agents)] = output.pop(actions_key, None)
            if len(output) > 0:
                append_dict2dict(output, logging_dict)
        logging_dict[actions_key] = actions

        if do_log:
            self.logger.log(logging_dict)

        return actions

    def _preprocess(self, obs):
        obs = obs.flatten()
        return obs
