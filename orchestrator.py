import numpy as np
import multiprocessing as mp

from logger import RunLogger
from custom import CoopsMetric, BatchSumAvgMetric, BatchAvgMetric, ActionMap, PolicyViaTime
from config import Config
from agents.q_attention import Agent
from elo_systems import MeanEloI
from utils import indices_except, append_dict2dict


class Orchestrator:

    def __init__(self, o_space: int, a_space: int, cfg: Config, queue: mp.Queue, name: str):
        self.cfg = cfg
        self.agents = [Agent(id=i, o_space=o_space, a_space=a_space, cfg=cfg) for i in range(cfg.players)]
        self.mean_elo = MeanEloI(cfg.players)

        self.logger = RunLogger(queue)
        metrics = [
            ActionMap('acts', cfg.players, [agent.label for agent in self.agents]),
            #PolicyViaTime('pvt', cfg.players, [agent.label for agent in self.agents]),
            CoopsMetric('acts', name)
        ]
        for agent in self.agents:
            agent.set_logger(self.logger)
            # metrics.append(ModelArt(f'{agent.agent_label}_model'))
            #metrics.append(BatchSumAvgMetric(f'{agent.label}_reward', 10, epoch_counter=False))
            #metrics.append(BatchAvgMetric(f'{agent.label}_elo', 10, epoch_counter=True))
            #metrics.append(BatchAvgMetric(f'{agent.label}_loss', 10, epoch_counter=False))
            # metrics.append(Metric(f'{agent.agent_label}_eps', epoch_counter=False, log_on_eval=False))

        self.logger.init(name, True, *metrics)
        self.logger.param(cfg.as_dict())

    def __del__(self):
        self.logger.deinit()

    def set_mode(self, train: bool = True):
        self.logger.set_mode(train)
        for agent in self.agents:
            agent.set_mode(train)

    def _make_act(self, obs):
        logging_dict = {}
        actions = np.zeros((2, len(self.agents), len(self.agents)), dtype=np.int32)
        actions_key = self.cfg.actions_key
        for agent_id, (agent, agent_obs) in enumerate(zip(self.agents, self._preprocess(obs))):
            output = agent.act(agent_obs) if agent.train else agent.inference(agent_obs)
            actions[:, agent_id, indices_except(agent_id, self.agents)] = output.pop(actions_key, None)
            if len(output) > 0:
                append_dict2dict(output, logging_dict)
        logging_dict[actions_key] = actions
        self.logger.log(logging_dict)

        return actions

    def act(self, obs):
        return self._make_act(obs)

    def inference(self, obs, episode):
        return self._make_act(obs)

    def rewarding(self, rewards, next_obs, last: bool):
        elo = self.mean_elo.step(rewards)

        for i, _ in enumerate(elo):
            self.logger.log({f'{self.agents[i].label}_elo': elo[i]})

        for agent, reward, agent_obs in zip(self.agents, rewards, self._preprocess(next_obs)):
            agent.rewarding(reward, agent_obs, last)

    def _preprocess(self, obs):
        return [np.concatenate((obs[i:], obs[:i])).reshape(-1)
                for i, _ in enumerate(self.agents)]
