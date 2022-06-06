import numpy as np
import multiprocessing as mp

import torch

from logger import RunLogger
from custom import ActionMap, SumArtifact, EMAArtifact
from config import Config
# from elo_systems import MeanElo
from agents import get_agent


class Orchestrator:

    def __init__(self, o_space: int, a_space: int, cfg: Config, name: str, queue: mp.Queue):
        self.cfg = cfg
        # self.mean_elo = MeanElo(cfg.players)
        self.train = None

        self.agents = [get_agent(cfg.agents[i], id=i, o_space=o_space, a_space=a_space, cfg=cfg)
                       for i in range(cfg.players)]
        self.eval_agents = [get_agent(cfg.eval_agents[i], id=i + cfg.players, o_space=o_space, a_space=a_space, cfg=cfg)
                            for i in range(cfg.eval_players)]

        is_ond = False
        if 'rps' in cfg.environment:
            labelsx = ['камень', 'бумага', 'ножницы']
        elif 'ond' in cfg.environment:
            labelsx = [agent.label for agent in self.agents]
            is_ond = True
        else:
            labelsx = ['кооперация', 'состязание']
        labelsy = [agent.label for agent in self.agents]

        metrics = (
            ActionMap(cfg.actions_key, labelsy, labelsx, is_ond, log_on_train=False),
            # # SumArtifact(cfg.reward_key, labelsy, name + '2', log_on_train=False, is_global=True),
            # # SumArtifact(cfg.reward_key, labelsy, name, log_on_eval=False, is_global=True),
            EMAArtifact(cfg.reward_key, labelsy, name, log_on_train=False, is_global=True),
            EMAArtifact(cfg.reward_key, labelsy, log_on_eval=False)
        )
        for agent in self.agents + self.eval_agents:
            metrics += agent.get_metrics()

        self.logger = RunLogger(queue, name, metrics)
        self.logger.param(cfg.as_dict())
        for agent in self.agents + self.eval_agents:
            agent.set_logger(self.logger)

        self._changed = [0]

        self.messages = None

    def __del__(self):
        self.logger.deinit()

    def reset(self, train: bool):
        self.train = train
        self.logger.set_mode(train)
        for agent in self.agents:
            agent.reset(train=train)
        # self.mean_elo.reset()

        if self.cfg.enable_eval_agents and not train:
            for i in range(self.cfg.eval_players):
                self.eval_agents[i].reset(train=True)
                self.agents[i + 1] = self.eval_agents[i]

    def negotiation(self):
        self.messages = [agent.message for agent in self.agents if agent.negotiable]
        for step in range(self.cfg.n_steps):
            self.messages = [agent.negotiate(self.messages, step) for agent in self.agents if agent.negotiable]

    def act(self, obs):
        obs = self._preprocess(obs)
        if self.cfg.enable_negotiation:
            actions = []
            for i, agent in enumerate(self.agents):
                if agent.negotiable:
                    messages = []
                    for j, message in enumerate(self.messages):
                        messages.append(message)
                        if i != j:
                            messages[-1] = messages[-1].detach()
                    actions.append(agent.act(obs, messages))
                else:
                    actions.append(agent.act(obs))
        else:
            actions = [agent.act(obs) for agent in self.agents]
        self.logger.log({self.cfg.actions_key: actions})
        return actions

    def rewarding(self, rewards, next_obs, last):
        next_obs = self._preprocess(next_obs)
        result = False
        for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
            if self.cfg.enable_negotiation:
                if agent.negotiable:
                    messages = []
                    for j, message in enumerate(self.messages):
                        messages.append(message)
                        if i != j:
                            messages[-1] = messages[-1].detach()
                    res = agent.rewarding(reward, next_obs, last, messages)
                else:
                    res = agent.rewarding(reward, next_obs, last)
            else:
                res = agent.rewarding(reward, next_obs, last)
            if not self.train and res is not None and res:
                result = True

        self.logger.log({self.cfg.reward_key: rewards})

        if not self.train:
            self._changed[-1] += 1
            if last:
                self.logger.call('ema_plots', self._changed[:-1] if len(self._changed) > 1 else None)
            if result or last:
                self._changed.append(self._changed[-1])
                self.logger.call('action_map')

        # if not self.train:
        #     elos = self.mean_elo.step(rewards)
        #     for agent, elo in zip(self.agents, elos):
        #         self.logger.log({agent.elo_key: elo})

    def _preprocess(self, obs):
        obs = obs.flatten()
        return obs
