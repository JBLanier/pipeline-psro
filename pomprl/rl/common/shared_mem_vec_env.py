"""
An interface for asynchronous vectorized environments.
"""

import multiprocessing as mp
import numpy as np
import ctypes
import gym
from collections import OrderedDict

from rllib.env.vector_env import VectorEnv

_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def obs_space_info(obs_space):
    """
    Get dict-structured information about a gym.Space.
    Returns:
      A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict)
        subspaces = obs_space.spaces
    elif isinstance(obs_space, gym.spaces.Tuple):
        assert isinstance(obs_space.spaces, tuple)
        subspaces = {i: obs_space.spaces[i] for i in range(len(obs_space.spaces))}
    else:
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


class ShmemVecEnv(VectorEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    """An environment that supports batch evaluation.

    Subclasses must define the following attributes:

    Attributes:
        action_space (gym.Space): Action space of individual envs.
        observation_space (gym.Space): Observation space of individual envs.
        num_envs (int): Number of envs in this vector env.
    """

    def __init__(self, make_env_fns, context='spawn'):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = mp.get_context(context)

        self.num_envs = len(make_env_fns)

        print('Creating dummy env object to get spaces')
        dummy = make_env_fns[0]()
        self.observation_space, self.action_space = dummy.observation_space, dummy.action_space
        self.agent_ids = dummy.all_possible_agent_ids
        dummy.close()
        del dummy

        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(self.observation_space)

        self.obs_bufs = {}
        self.ids_avail_vals = {}

        for agent_id in self.agent_ids:
            self.obs_bufs[agent_id] = [
                {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
                for _ in make_env_fns]
            self.ids_avail_vals[agent_id] = [ctx.Value(typecode_or_type=_NP_TO_CT[np.np.bool])
                                             for _ in make_env_fns]

        self.parent_pipes = []
        self.procs = []
        # with clear_mpi_env_vars():
        for env_idx, env_fn in enumerate(make_env_fns):
            wrapped_fn = CloudpickleWrapper(env_fn)
            parent_pipe, child_pipe = ctx.Pipe()

            obs_bufs = {agent_id: self.obs_bufs[agent_id][env_idx] for agent_id in self.agent_ids}
            ids_avail_vals = {agent_id: self.ids_avail_vals[agent_id][env_idx] for agent_id in self.agent_ids}

            proc = ctx.Process(target=_subproc_worker,
                        args=(child_pipe, parent_pipe, wrapped_fn, obs_bufs, ids_avail_vals, self.obs_shapes, self.obs_dtypes, self.obs_keys))
            proc.daemon = True
            self.procs.append(proc)
            self.parent_pipes.append(parent_pipe)
            proc.start()
            child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def vector_reset(self):
        """Resets all environments.

        Returns:
            obs (list): Vector of observations from each environment.
        """

        if self.waiting_step:
            print('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))

        [pipe.recv() for pipe in self.parent_pipes]
        return self._decode_obses()

    def reset_at(self, index):
        """Resets a single environment.

        Returns:
            obs (obj): Observations from the resetted environment.
        """

        if self.waiting_step:
            raise ValueError('Called reset() while waiting for the step to complete')

        pipe = self.parent_pipes[index]
        pipe.send(('reset', None))
        pipe.recv()
        return self._decode_single_observation(env_idx=index)

    def vector_step(self, actions):
        """Vectorized step.

        Arguments:
            actions (list): Actions for each env.

        Returns:
            obs (list): New observations for each env.
            rewards (list): Reward values for each env.
            dones (list): Done values for each env.
            infos (list): Info values for each env.
        """
        self.step_async(actions=actions)
        return self.step_wait()

    def get_unwrapped(self):
        """Returns the underlying env instances."""
        raise NotImplementedError

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))
        self.waiting_step = True

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        obs, rews, dones, infos = zip(*outs)
        return self._decode_obses(), np.array(rews), np.array(dones), infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def _decode_single_observation(self, env_idx):
        obs = {}
        for agent_id in self.agent_ids:
            agent_id_available = self.ids_avail_bufs[agent_id][env_idx]
            if agent_id_available:
                for k in self.obs_keys:
                    buf = self.obs_bufs[agent_id][env_idx][k]
                    o = np.frombuffer(buf.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k])
                    obs[agent_id][k] = np.array(o)
        return obs

    def _decode_obses(self):
        result = []
        for env_idx in range(self.num_envs):
            result.append(self._decode_single_observation(env_idx=env_idx))
        return result


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, ids_avail_vals, obs_shapes, obs_dtypes, keys):
    """
    Control a single environment instance using IPC and
    shared memory.
    """

    env = env_fn_wrapper.x()
    agent_ids = env.all_possible_agent_ids
    parent_pipe.close()

    def _write_obs(dict_obs):
        obs_agent_ids = dict_obs.keys
        for agent_id in agent_ids:

            if agent_id in obs_agent_ids:
                ids_avail_vals[agent_id].value = True

                for k in keys:
                    dst = obs_bufs[agent_id][k].get_obj()
                    dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
                    np.copyto(dst_np, dict_obs[agent_id][k])

            else:
                ids_avail_vals[agent_id].value = False

    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                pipe.send(_write_obs(env.reset()))
            elif cmd == 'step':
                obs, rewards, dones, infos = env.step(data)
                pipe.send((_write_obs(obs), rewards, dones, infos))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()