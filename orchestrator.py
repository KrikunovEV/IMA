import torch
import multiprocessing as mp

from logger import RunLogger, Metric, ModelArt
from custom import BatchSumAvgMetric, ActionMap, PolicyViaTime, CoopsMetric, BatchAvgMetric
from config import Config
from agent import Agent
from elo_systems import MeanElo


class Orchestrator:

    def __init__(self, o_space: int, a_space: int, cfg: Config, queue: mp.Queue, name: str):
        self.cfg = cfg
        self.agents = [Agent(id=i, o_space=o_space, a_space=a_space, cfg=cfg) for i in range(cfg.players)]
        self.mean_elo = MeanElo(cfg.players)
        self.m = dict()

        self.logger = RunLogger(queue)
        metrics = [
            # ActionMap('acts', cfg.players, [agent.label for agent in self.agents]),
            # PolicyViaTime('pvt', cfg.players, [agent.label for agent in self.agents]),
            CoopsMetric('acts', name)
        ]
        for agent in self.agents:
            agent.set_logger(self.logger)
            # metrics.append(ModelArt(f'{agent.agent_label}_model'))
            # metrics.append(BatchSumAvgMetric(f'{agent.label}_reward', 10, epoch_counter=False))
            # metrics.append(BatchAvgMetric(f'{agent.label}_elo', 10, epoch_counter=True))
            # metrics.append(Metric(f'{agent.agent_label}_loss', epoch_counter=False))
            # metrics.append(Metric(f'{agent.agent_label}_eps', epoch_counter=False, log_on_eval=False))
        self.logger.init(name, True, *metrics)
        self.logger.param(cfg.as_dict())

    def __del__(self):
        self.logger.deinit()

    def set_mode(self, train: bool = True):
        self.logger.set_mode(train)
        for agent in self.agents:
            agent.set_mode(train)

    def reset_memory(self):
        self.mean_elo.reset()
        self.logger.all()
        for agent in self.agents:
            agent.reset_memory()

    def negotiation(self):
        q_all = dict()
        for agent in self.agents:
            for key, value in agent.message().items():
                q_all[key] = value

        self.m = dict()
        for a_id, value in q_all.items():
            my_q = value
            q = []
            for a_id2, value2 in q_all.items():
                if a_id == a_id2:
                    continue
                q.append(value2.detach())
            a_id = int(a_id)
            self.m[a_id] = self.agents[a_id].negotiate(my_q, torch.stack(q))

    def act(self, obs):
        obs = torch.Tensor(obs).view(-1).unsqueeze(0).to(self.cfg.device)
        actions, a_policies, d_policies = [], [], []
        for agent in self.agents:
            m = None
            if agent.id in self.m:
                m = self.m[agent.id]
            acts, ap, dp = agent.act(obs, m)
            actions.append(acts)
            a_policies.append(ap.detach().cpu().numpy())
            d_policies.append(dp.detach().cpu().numpy())

        self.logger.log({'acts': actions, 'pvt': (a_policies, d_policies)})
        return actions

    def rewarding(self, rewards, last):
        elo = self.mean_elo.step(rewards)
        for i in range(len(elo)):
            self.logger.log({f'{self.agents[i].label}_elo': elo[i]})

        for agent, reward in zip(self.agents, rewards):
            agent.rewarding(reward, last)

    def learn(self):
        for agent in self.agents:
            agent.learn()
