import numpy as np
import gym
from collections import OrderedDict
import logging
from pomprl.rl.envs.stratego.stratego_multiagent_env import STRATEGO_ENV
from pomprl.rl.common.cloud_storage import upload_file, maybe_download_object, get_default_path_on_disk_for_minio_key
from pomprl.util import datetime_str
import os
import binascii

logger = logging.getLogger(__name__)

def numpy_unpack_obs(obs, space, preprocessor):
    """Unpack a flattened Dict or Tuple observation array/ndarray.

    Arguments:
        obs: The flattened observation tensor
        space: The original space prior to flattening
        preprocessor: The preprocessor used to create this observation
    """

    if (isinstance(space, gym.spaces.Dict)
            or isinstance(space, gym.spaces.Tuple)):
        prep = preprocessor
        if len(obs.shape) != 2 or obs.shape[1] != prep.shape[0]:
            raise ValueError(
                "Expected flattened obs shape of [None, {}], got {}".format(
                    prep.shape[0], obs.shape))
        assert len(prep.preprocessors) == len(space.spaces), \
            (len(prep.preprocessors) == len(space.spaces))
        offset = 0
        if isinstance(space, gym.spaces.Tuple):
            u = []
            for p, v in zip(prep.preprocessors, space.spaces):
                obs_slice = obs[:, offset:offset + p.size]
                offset += p.size
                u.append(
                    numpy_unpack_obs(
                        np.reshape(obs_slice, [-1] + list(p.shape)),
                        v,
                        prep))
        else:
            u = OrderedDict()
            for p, (k, v) in zip(prep.preprocessors, space.spaces.items()):
                obs_slice = obs[:, offset:offset + p.size]
                offset += p.size
                u[k] = numpy_unpack_obs(
                    np.reshape(obs_slice, [-1] + list(p.shape)),
                    v,
                    prep)
        return u
    else:
        return obs


def get_redo_sample_if_game_result_was_invalid_worker_callback(on_sample_end_callback):

    def redo_sample_if_game_result_was_invalid_worker_callback(info, first_call=True):
        worker = info['worker']
        samples = info['samples']

        resampling_needed = False
        for policy_id, batch in samples.policy_batches.items():
            assert batch['dones'][-1], "the last step in the batch wasn't marked \'done\'"

            if batch['infos'][-1]['game_result_was_invalid']:
                resampling_needed = True
                logger.debug("batch was bad (length of {})".format(samples.count))

                # remove this batch from the input reader's metrics as well (this is kinda hacky)
                with worker.input_reader.metrics_queue.not_empty:
                    worker.input_reader.metrics_queue.queue.pop()

                break

        if resampling_needed:
            logger.debug("resampling episode due to invalid game result.")
            new_samples = worker.sample()
            samples.policy_batches = new_samples.policy_batches
            samples.count = new_samples.count

            info['samples'] = samples
            redo_sample_if_game_result_was_invalid_worker_callback(info, first_call=False)
        elif first_call:
            logger.debug("batch was good (length of {})".format(samples.count))
            if on_sample_end_callback:
                on_sample_end_callback(info)

        #TODO (JB) remove this part, its just verification
        for policy_id, batch in samples.policy_batches.items():

            if batch['infos'][-1]['game_result_was_invalid']:
                break

            for i, info in enumerate(batch['infos']):
                if "game_result_was_invalid" in info.keys() and info['game_result_was_invalid']:
                    assert False, "This super shouldnt have happened (at step {})".format(i)

    return redo_sample_if_game_result_was_invalid_worker_callback


def set_policies_to_train_in_trainer(trainer, new_policies_to_train):
    def worker_assign_policies_to_train(worker):
        worker.policies_to_train = new_policies_to_train

    print("NEW TRAINABLE POLICIES ARE", new_policies_to_train)
    trainer.workers.foreach_worker(worker_assign_policies_to_train)
    trainer.optimizer.policies = dict(trainer.workers.local_worker().foreach_trainable_policy(lambda p, i: (i, p)))


def get_short_trial_id(trial):
    if "env" in trial.config:
        env = trial.config["env"]

        if env == STRATEGO_ENV:
            identifier = "{}_{}_{}".format(
                trial.trainable_name[:8],
                trial.config["env_config"]['version'],
                trial.config["env_config"]['observation_mode'][:10])
        else:
            if isinstance(env, type):
                env = env.__name__
            identifier = "{}_{}".format(trial.trainable_name, env)
    else:
        identifier = trial.trainable_name

    if trial.experiment_tag:
        identifier += "_" + ''.join(c for c in trial.experiment_tag[:3] if c.isdigit())

    identifier += "_" + trial.trial_id

    return identifier.replace("/", "_")


def get_random_policy_object_key(base_experiment_name, full_experiment_name, tag: str = None):

    file_name = "{}_{}.policy".format(datetime_str(), binascii.b2a_hex(os.urandom(6)).decode("utf-8"))

    if tag is None:
        return os.path.join(base_experiment_name, full_experiment_name, file_name)
    else:
        return os.path.join(base_experiment_name, full_experiment_name, tag, file_name)


def save_model_weights_to_disk_tmp_and_minio(minio_client, bucket_name, policy_to_save, policy_key, new_object_key):
    disk_path = get_default_path_on_disk_for_minio_key(object_name=new_object_key)
    policy_to_save.save_model_weights(disk_path, remove_scope_prefix=policy_key)
    upload_file(minio_client, bucket_name=bucket_name, object_key=new_object_key, local_source_path=disk_path)


def load_model_weights_from_disk_tmp_or_minio(minio_client, bucket_name, policy_to_set, policy_key, object_key):
    maybe_download_object(minio_client, bucket_name=bucket_name, object_name=object_key)
    disk_path = get_default_path_on_disk_for_minio_key(object_name=object_key)
    policy_to_set.load_model_weights(disk_path, add_scope_prefix=policy_key)