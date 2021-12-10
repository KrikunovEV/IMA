import torch
import multiprocessing as mp

from logger import RunLogger
from custom import CoopsMetric, BatchSumAvgMetric, BatchAvgMetric
from config import Config
from agents.q_attention import Agent
from elo_systems import MeanElo


class Orchestrator:

    def __init__(self, o_space: int, a_space: int, cfg: Config, queue: mp.Queue, name: str):
        self.cfg = cfg
        self.agents = [Agent(id=i, o_space=o_space, a_space=a_space, cfg=cfg) for i in range(cfg.players)]
        self.mean_elo = MeanElo(cfg.players)

        self.logger = RunLogger(queue)
        metrics = [
            # ActionMap('acts', cfg.players, [agent.label for agent in self.agents]),
            # PolicyViaTime('pvt', cfg.players, [agent.label for agent in self.agents]),
            CoopsMetric('acts', name)
        ]
        for agent in self.agents:
            agent.set_logger(self.logger)
            # metrics.append(ModelArt(f'{agent.agent_label}_model'))
            metrics.append(BatchSumAvgMetric(f'{agent.label}_reward', 10, epoch_counter=False))
            metrics.append(BatchAvgMetric(f'{agent.label}_elo', 10, epoch_counter=True))
            metrics.append(BatchAvgMetric(f'{agent.label}_loss', 10, epoch_counter=False))
            # metrics.append(Metric(f'{agent.agent_label}_eps', epoch_counter=False, log_on_eval=False))
        self.logger.init(name, True, *metrics)
        self.logger.param(cfg.as_dict())

    def __del__(self):
        self.logger.deinit()

    def set_mode(self, train: bool = True):
        self.logger.set_mode(train)
        for agent in self.agents:
            agent.set_mode(train)

    def act(self, obs):
        local_obs = self._preprocess(obs)
        actions, o_policies, d_policies = [], [], []
        for agent, agent_obs in zip(self.agents, local_obs):
            output = agent.act(agent_obs)
            actions.append(output['acts'])
            o_policies.append(output['policies'][0].detach().cpu().numpy())
            d_policies.append(output['policies'][1].detach().cpu().numpy())
        self.logger.log({'acts': actions, 'pvt': (o_policies, d_policies)})
        return actions

    def rewarding(self, rewards, next_obs, last: bool):
        elo = self.mean_elo.step(rewards)
        for i in range(len(elo)):
            self.logger.log({f'{self.agents[i].label}_elo': elo[i]})

        local_next_obs = self._preprocess(next_obs)
        for agent, reward, agent_obs in zip(self.agents, rewards, local_next_obs):
            agent.rewarding(reward, agent_obs, last)

    def inference(self, obs, episode):
        local_obs = self._preprocess(obs)
        actions = []
        for agent, agent_obs in zip(self.agents, local_obs):
            output = agent.inference(agent_obs)
            actions.append(output['acts'])

        if episode >= self.cfg.window * 2:
            # first 'window' times are random, second 'window' times fill history by following right policy
            self.logger.log({'acts': actions})

        return actions

    def _preprocess(self, obs):
        obs = torch.from_numpy(obs)
        local_obs = []
        for i, agent in enumerate(self.agents):
            _obs = torch.concat((obs[i:], obs[:i])).view(-1)
            local_obs.append(_obs)
        return local_obs
