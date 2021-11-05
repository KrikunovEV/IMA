import torch
import multiprocessing as mp

from logger import RunLogger, Metric, ModelArt
from custom import SumMetric, ActionMap
from config import Config
from agent import Agent


class Orchestrator:

    def __init__(self, o_space: int, a_space: int, cfg: Config, queue: mp.Queue, name: str):
        self.cfg = cfg
        self.agents = [Agent(id=i, o_space=o_space, a_space=a_space, cfg=cfg) for i in range(cfg.players)]

        self.logger = RunLogger(queue)
        metrics = [ActionMap('am', cfg.players, [agent.agent_label for agent in self.agents])]
        for agent in self.agents:
            agent.set_logger(self.logger)
            metrics.append(ModelArt(f'{agent.agent_label}_model'))
            metrics.append(SumMetric(f'{agent.agent_label}_reward', epoch_counter=False))
            metrics.append(Metric(f'{agent.agent_label}_loss', epoch_counter=False))
            metrics.append(Metric(f'{agent.agent_label}_eps', epoch_counter=False, log_on_eval=False))
        self.logger.init(name, True, *metrics)
        self.logger.param(cfg.as_dict())

    def __del__(self):
        self.logger.deinit()

    def set_mode(self, train: bool = True):
        self.logger.set_mode(train)
        for agent in self.agents:
            agent.set_mode(train)

    def reset_memory(self):
        for agent in self.agents:
            agent.reset_memory()

    def act(self, obs):
        obs = torch.Tensor(obs).view(-1).unsqueeze(0).to(self.cfg.device)
        actions = [agent.act(obs) for agent in self.agents]
        self.logger.log({'am': actions})
        return actions

    def rewarding(self, rewards):
        for agent, reward in zip(self.agents, rewards):
            agent.rewarding(reward)

    def learn(self):
        for agent in self.agents:
            agent.learn()
