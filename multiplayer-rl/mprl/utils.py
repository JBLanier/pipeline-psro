import os
import multiprocessing as mp
import numpy as np
import copy
from datetime import datetime
import json
import yaml
import numbers


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


def load_training_data_from_file(name):
    return np.load('{}.npy'.format(name))


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def policy_with_dirichlet_noise(policy, valid_actions_mask, noise_proportion):
    sum_valid_actions = np.sum(valid_actions_mask)
    dirichlet_noise_valid_actions = np.random.dirichlet(alpha=[1. / sum_valid_actions] * sum_valid_actions)
    dirichlet_noise_valid_actions = iter(dirichlet_noise_valid_actions)
    dirichlet_noise = np.asarray(
        [next(dirichlet_noise_valid_actions) if is_valid else 0. for is_valid in valid_actions_mask])

    assert np.isclose(np.sum(dirichlet_noise), 1.0)

    new_policy = (1. - noise_proportion) * policy + noise_proportion * dirichlet_noise
    new_policy /= np.sum(new_policy)

    return new_policy


pool_worker_index = None


def worker_index_init(queue=None, index_val=None):
    global pool_worker_index

    if queue is not None:
        pool_worker_index = queue.get()
    else:
        pool_worker_index = index_val


def create_worker_process_pool_with_ids(num_workers):
    # Creates a normal process pool except each process has a unique global variable 'idx' with its worker index
    # Each worker can use 'idx' to know who it is.

    ids = list(range(num_workers))
    manager = mp.Manager()
    id_queue = manager.Queue()

    for i in ids:
        id_queue.put(i)

    return mp.Pool(processes=num_workers, initializer=worker_index_init, initargs=(id_queue,))


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def with_base_config(base_config, extra_config):
    """Returns the given config dict merged with a base agent conf."""

    config = copy.deepcopy(base_config)
    config.update(extra_config)
    return config


def datetime_str():
    # format is ok for file/directory names
    date_string = datetime.now().strftime("%I.%M.%S%p_%b-%d-%Y")
    return date_string


def softmax(x, temperature = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    x: ND-Array. Probably should be floats.
    temperature (optional): float parameter, used as a divisor
        prior to exponentiation. Default = 1.0 Can be [0, inf)
        Temp near 0 approaches argmax, near inf approaches uniform dist
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.

    https://nolanbconaway.github.io/blog/2017/softmax-numpy
    """

    # make X at least 2d
    y = np.atleast_2d(x)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y / float(temperature)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(x.shape) == 1: p = p.flatten()

    return p


class _SafeFallbackEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(_SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return self.nan_str

            if (type(value).__module__ == np.__name__
                    and isinstance(value, np.ndarray)):
                return value.tolist()

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(_SafeFallbackEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_print(result):
    result = result.copy()
    result.update(config=None)  # drop config from pretty print
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=_SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)


def with_updates(base_dict: dict, updates_dict: dict):
    out = base_dict.copy()
    out.update(updates_dict)
    return out