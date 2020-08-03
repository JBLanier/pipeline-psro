import ray
import numpy as np
from trueskill import Rating, quality_1vs1, rate_1vs1
import random
from pomprl.rl.common.cloud_storage import connect_minio_client, upload_file, get_default_path_on_disk_for_minio_key, maybe_download_object
from pomprl.rl.common.util import save_model_weights_to_disk_tmp_and_minio, load_model_weights_from_disk_tmp_or_minio
from dill import load, dump
from pomprl.util import ensure_dir
from collections import OrderedDict
from copy import deepcopy
import os
import csv
import time
from itertools import repeat

MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

POLICY_KEY = 'policy_key'
SKILL_RANKING = 'skill_ranking'
STEPS_TRAINED = 'steps_trained'
TIME_CREATED = 'time_created'
GAMES_RANKED_ON = 'games_ranked_on'
WALL_TIME_SINCE_START = 'wall_time_since_start'
IS_POLICY_MIXTURE = 'is_policy_mixture'

RECORD_CSV_FIELDS = [WALL_TIME_SINCE_START, SKILL_RANKING, POLICY_KEY, STEPS_TRAINED, TIME_CREATED, GAMES_RANKED_ON,
                     IS_POLICY_MIXTURE]

DEFAULT_RECORD_ENTRY_INTERVAL_SECONDS = 60 * 20


@ray.remote(num_cpus=1)
class PolicyCatalog:

    def __init__(self,
                 cache_size=0,
                 record_file_path=None,
                 new_record_entry_every_n_seconds=DEFAULT_RECORD_ENTRY_INTERVAL_SECONDS,
                 extra_data_keys=None):

        self.catalog = {}
        self.minio_client = connect_minio_client(endpoint=MINIO_ENDPOINT,
                                        access_key=MINIO_ACCESS_KEY,
                                        secret_key=MINIO_SECRET_KEY)
        self.bucket_name = BUCKET_NAME

        self.cache = OrderedDict()
        self.cache_size = cache_size

        self.record_file_path = record_file_path
        self.new_record_entry_every_n_seconds = new_record_entry_every_n_seconds
        self.start_time = time.time()
        self.last_record_entry_time = self.start_time
        self.extra_data_keys = extra_data_keys or []

    def add_to_cache(self, policy_key, weights):

        if self.cache_size > 0:
            if policy_key in self.cache:
                self.cache[policy_key] = weights
            else:
                while len(self.cache) >= self.cache_size:
                    # pop oldest item from cache
                    self.cache.popitem(last=False)
                self.cache[policy_key] = weights

    def get_from_cache(self, policy_key):
        if policy_key in self.cache:
            return self.cache[policy_key]
        return None

    def _get_catalog_subset(self, exclude_keys):
        if exclude_keys is None:
            exclude_keys = []

        tmp_catalog = self.catalog.copy()
        for exclude_key in exclude_keys:
            if exclude_key in tmp_catalog:
                del tmp_catalog[exclude_key]
        return tmp_catalog

    def get_size(self):
        return len(self.catalog)

    def submit_new_policy(self, policy_file_key, policy_weights, steps_trained, extra_data: dict = None):
        self.add_to_cache(policy_key=policy_file_key, weights=policy_weights)

        # save policy weights to disk and cloud

        disk_path = get_default_path_on_disk_for_minio_key(object_name=policy_file_key)
        ensure_dir(disk_path)
        with open(disk_path, "wb") as dill_file:
            dump(obj=policy_weights, file=dill_file)

        upload_file(self.minio_client, bucket_name=self.bucket_name, object_key=policy_file_key, local_source_path=disk_path)

        if policy_file_key not in self.catalog:
            ranked_games_played = 0
            skill_ranking = Rating()
            policy_mixture = None
            data = {STEPS_TRAINED: steps_trained,
                    TIME_CREATED: time.time() - self.start_time}

            if extra_data is not None:
                data.update(extra_data)
            self.catalog[policy_file_key] = (skill_ranking, data, ranked_games_played, policy_mixture)

    def submit_policy_mixture(self, mixture_key, policy_file_keys_to_probs_dict, steps_trained, extra_data: dict = None):

        assert np.isclose(1.0, sum(prob for prob in policy_file_keys_to_probs_dict.values()))
        for policy_key in policy_file_keys_to_probs_dict.keys():
            assert policy_key in self.catalog

        if mixture_key not in self.catalog:
            ranked_games_played = 0
            skill_ranking = Rating()
            policy_mixture = policy_file_keys_to_probs_dict
            data = {STEPS_TRAINED: steps_trained,
                    TIME_CREATED: time.time() - self.start_time}

            if extra_data is not None:
                data.update(extra_data)
            self.catalog[mixture_key] = (skill_ranking, data, ranked_games_played, policy_mixture)


    def get_random_policy_file_key(self, exclude_keys=None):
        tmp_catalog = self._get_catalog_subset(exclude_keys=exclude_keys)
        selected_key = random.sample(tmp_catalog.keys(), k=1)[0]
        selected_key = self.sample_policy_key_if_key_is_for_mixture(selected_key)
        return selected_key

    def get_policy_file_key_weighted_by_skill_similarity(self, policy_to_match_with, exclude_keys=None):
        if exclude_keys is None:
            exclude_keys = []

        exclude_keys.append(policy_to_match_with)
        policy_skill_rating, data, games_ranked_on, policy_mixture = self.catalog[policy_to_match_with]

        tmp_catalog = self._get_catalog_subset(exclude_keys=exclude_keys)

        all_policy_keys, policy_vals = zip(*tmp_catalog.items())
        all_policy_skills = [v[0] for v in policy_vals]
        weights = [quality_1vs1(rating1=policy_skill_rating, rating2=other) for other in all_policy_skills]
        weights = np.asarray(weights)
        weights /= sum(weights)

        selected_policy_key = np.random.choice(all_policy_keys, p=weights)

        assert selected_policy_key != policy_to_match_with

        selected_policy_key = self.sample_policy_key_if_key_is_for_mixture(selected_policy_key)

        return selected_policy_key

    def sample_policy_key_if_key_is_for_mixture(self, policy_key):
        policy_skill_rating, data, games_ranked_on, policy_mixture = self.catalog[policy_key]
        if policy_mixture is None:
            return policy_key
        else:
            return self.sample_policy_key_if_key_is_for_mixture(
                policy_key=np.random.choice(policy_mixture.keys(), p=policy_mixture.values()))

    @ray.method(num_return_vals=2)
    def get_policy_weights_weighted_by_skill_similarity(self, policy_to_match_with, exclude_keys=None):

        new_policy_file_key = self.get_policy_file_key_weighted_by_skill_similarity(
            policy_to_match_with=policy_to_match_with, exclude_keys=exclude_keys)

        weights = self.get_weights_by_key(new_policy_file_key)

        return weights, new_policy_file_key

    @ray.method(num_return_vals=2)
    def get_random_policy_weights(self, exclude_keys=None):

        new_policy_file_key = self.get_random_policy_file_key(exclude_keys=exclude_keys)

        weights = self.get_weights_by_key(new_policy_file_key)

        return weights, new_policy_file_key

    def get_weights_by_key(self, policy_key):
        weights = self.get_from_cache(policy_key=policy_key)

        if weights is None:
            load_file_path, _ = maybe_download_object(minio_client=self.minio_client, bucket_name=self.bucket_name,
                                                      object_name=policy_key)

            with open(load_file_path, "rb") as dill_file:
                weights = load(file=dill_file)

        return weights

    def submit_game_result_for_ranking(self, winner_file_key, loser_file_key, was_tie=False):
        winner_old_ranking, winner_data, winner_games_played, winner_mixture = self.catalog[winner_file_key]
        loser_old_ranking, loser_data, loser_games_played, loser_mixture = self.catalog[loser_file_key]

        winner_new_ranking, loser_new_ranking = rate_1vs1(
            rating1=winner_old_ranking,
            rating2=loser_old_ranking,
            drawn=was_tie)

        new_winner_games_played = winner_games_played + 1
        new_loser_games_played = loser_games_played + 1

        self.catalog[winner_file_key] = (winner_new_ranking, winner_data, new_winner_games_played, winner_mixture)
        self.catalog[loser_file_key] = (loser_new_ranking, loser_data, new_loser_games_played, loser_mixture)

        now = time.time()
        if now - self.last_record_entry_time >= self.new_record_entry_every_n_seconds:
            self._add_record_entry()
            self.last_record_entry_time = now

    def get_skill_ranking(self, policy_file_key):
        skill_ranking, data, games_ranked_on, policy_mixture = self.catalog[policy_file_key]
        return skill_ranking.mu

    def _add_record_entry(self):
        if self.record_file_path:
            if not os.path.exists(self.record_file_path):
                ensure_dir(self.record_file_path)
                header_needs_write = True
            else:
                header_needs_write = False

            with open(self.record_file_path, 'a') as csv_file:
                writer = csv.DictWriter(f=csv_file, fieldnames=RECORD_CSV_FIELDS + self.extra_data_keys)
                if header_needs_write:
                    writer.writeheader()

                wall_time = time.time() - self.start_time

                for policy_key, vals in self.catalog.items():
                    skill_ranking, data, games_ranked_on, policy_mixture = vals

                    row = {
                        WALL_TIME_SINCE_START: wall_time,
                        POLICY_KEY: policy_key,
                        SKILL_RANKING: skill_ranking.mu,
                        STEPS_TRAINED: data[STEPS_TRAINED],
                        TIME_CREATED: data[TIME_CREATED],
                        GAMES_RANKED_ON: games_ranked_on,
                        IS_POLICY_MIXTURE: policy_mixture is not None}

                    row.update({k: data[k] for k in self.extra_data_keys})

                    writer.writerow(row)

    def get_all_keys(self):
        return list(self.catalog.keys())