import numbers
import wandb
from ray import tune
from ray.tune.util import flatten_dict
import ray


def make_dict_items_yaml_representable(repr_dict):

    for k, v in repr_dict.items():
        if type(v) is dict:
            make_dict_items_yaml_representable(v)
        if isinstance(v, tuple):
            repr_dict[k] = (str(elem) if not isinstance(elem, (int, float, complex, bool, str)) else elem for elem in v)
        if isinstance(v, list):
            repr_dict[k] = [str(elem) if not isinstance(elem, (int, float, complex, bool, str)) else elem for elem in v]
        elif not isinstance(v, (int, float, complex, bool, str)):
            repr_dict[k] = str(v)


@ray.remote(num_cpus=0)
class _WandbLoggerWorker():

    def __init__(self, trial_id, config):
        self._config = None

        wandb.init(name=trial_id, id=trial_id, **config.get("wandb", {}))

    def on_result(self, result):
        tmp = result.copy()

        config = tmp.get("config")
        if config and self._config is None:

            make_dict_items_yaml_representable(config)

            for k in config.keys():
                if wandb.config.get(k) is None:
                    wandb.config[k] = config[k]

            self._config = config

        for k in ["done", "config", "pid", "timestamp"]:
            if k in tmp:
                del tmp[k]
        metrics = {}
        for key, value in flatten_dict(tmp, delimiter="/").items():
            if not isinstance(value, numbers.Number):
                continue
            metrics[key] = value
        wandb.log(metrics, step=tmp.get("timesteps_total", None))

    def close(self):
        wandb.join()
        ray.actor.exit_actor()


class WandbLogger(tune.logger.Logger):
    """Pass WandbLogger to the loggers argument of tune.run

       tune.run("PG", loggers=[WandbLogger], config={
           "monitor": True,
           "wandb": {"project": "my-project-name"}}})
    """

    def _init(self):
        if self.trial.custom_trial_name is not None:
            trail_id = self.trial.custom_trial_name
        else:
            trail_id = self.trial.trial_id

        self._actual_wandb_logger = _WandbLoggerWorker.remote(trial_id=trail_id, config=self.config)

    def on_result(self, result):
        self._actual_wandb_logger.on_result.remote(result)

    def close(self):
        self._actual_wandb_logger.close.remote()
