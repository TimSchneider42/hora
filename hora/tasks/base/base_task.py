# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------
import abc
import gym
import multiprocessing.pool
import numpy as np
import sys
from abc import ABC
from gym import spaces
from typing import Dict, Any, Tuple, Iterable, Callable
from queue import Empty
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Env(ABC):
    observation_space: gym.Space
    action_space: gym.Space
    num_envs: int

    @abc.abstractmethod
    def step(
        self, actions: "torch.Tensor"
    ) -> Tuple[
        Dict[str, "torch.Tensor"], "torch.Tensor", "torch.Tensor", Dict[str, Any]
    ]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self) -> Dict[str, "torch.Tensor"]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @property
    def num_observations(self) -> int:
        """Get the number of observations in the environment."""
        return self.observation_space.shape[0]

    @property
    def num_actions(self) -> int:
        return self.action_space.shape[0]

    def close(self):
        pass


def _env_process(
    env_factory: Callable[[], Env],
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
):
    try:
        env = env_factory()
        output_queue.put((env.observation_space, env.action_space, env.num_envs))
        terminate = False
        while not terminate:
            cmd = input_queue.get()
            if cmd is None:
                terminate = True
            elif cmd == "reset":
                output_queue.put(env.reset())
            else:
                output_queue.put(env.step(cmd))
    except Exception as e:
        traceback.print_exc()
        output_queue.put((e, e.__traceback__))


class AsyncEnvGroup(Env):
    def __init__(self, env_factories: Iterable[Callable[[], Env]]):
        self._input_queues = [multiprocessing.Queue() for _ in env_factories]
        self._output_queues = [multiprocessing.Queue() for _ in env_factories]
        self._processes = [
            multiprocessing.Process(target=_env_process, args=(factory, iq, oq))
            for factory, iq, oq in zip(
                env_factories, self._input_queues, self._output_queues
            )
        ]
        for p in self._processes:
            p.start()
        res = self._get_res()
        self._num_envs = [r[2] for r in res]
        self._action_space = res[0][1]
        self._observation_space = res[0][0]
        self._weights = np.array(self._num_envs, dtype=np.float32)
        self._weights /= np.sum(self._weights)

    def _split_tensor(self, tensor: "torch.Tensor") -> Tuple["torch.Tensor"]:
        splits = []
        start = 0
        for num_envs in self._num_envs:
            end = start + num_envs
            splits.append(tensor[start:end])
            start = end
        return tuple(splits)

    def _join_tensors(self, tensors: Iterable["torch.Tensor"]) -> "torch.Tensor":
        import torch

        return torch.cat(list(tensors), dim=0)

    def _join_dicts(
        self, dicts: Iterable[Dict[str, "torch.Tensor"]]
    ) -> Dict[str, "torch.Tensor"]:
        return {k: self._join_tensors([r[k] for r in dicts]) for k in dicts[0].keys()}

    def _join_extras(self, extras: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        joined = {}
        for k in extras[0].keys():
            values = [r[k] for r in extras]
            if values[0].ndim > 0:
                joined[k] = self._join_tensors(values)
            else:
                joined[k] = torch.sum(
                    torch.tensor(values) * torch.from_numpy(self._weights)
                )
        return joined

    def step(
        self, actions: "torch.Tensor"
    ) -> Tuple[
        Dict[str, "torch.Tensor"], "torch.Tensor", "torch.Tensor", Dict[str, Any]
    ]:
        for q, a in zip(self._input_queues, self._split_tensor(actions)):
            q.put(a)
        obs, rew, reset, extra = zip(*self._get_res())
        return (
            self._join_dicts(obs),
            self._join_tensors(rew),
            self._join_tensors(reset),
            self._join_extras(extra),
        )

    def reset(self) -> Dict[str, "torch.Tensor"]:
        for q in self._input_queues:
            q.put("reset")
        return self._join_dicts(self._get_res())

    def _get_res(self):
        res = [None for _ in self._output_queues]
        while any(r is None for r in res):
            for i, r in enumerate(res):
                if r is None:
                    try:
                        res[i] = self._output_queues[i].get(timeout=0.1)
                        if isinstance(res[i], tuple) and isinstance(
                            res[i][0], Exception
                        ):
                            self.close()
                            raise res[i][0].with_traceback(res[i][1])
                    except Empty:
                        pass
        return res

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_envs(self):
        return sum(self._num_envs)

    def close(self):
        for q in self._input_queues:
            q.put(None)
        for p in self._processes:
            p.join()
