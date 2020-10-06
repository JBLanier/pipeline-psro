import ray
import numpy as np
import random
from mprl.utility_services.cloud_storage import connect_storage_client, upload_file, get_default_path_on_disk_for_minio_key, maybe_download_object
from mprl.rl.common.util import save_model_weights_to_disk_tmp_and_minio, load_model_weights_from_disk_tmp_or_minio
from mprl.rl.metanash.measurements import get_alpha_rank_pi, get_effective_population_diversity
from dill import load, dump
from mprl.utils import ensure_dir
from collections import OrderedDict
from copy import deepcopy
import os
import csv
import time
import plotly.express as px
import plotly
import pandas as pd

from itertools import repeat, chain, zip_longest

POLICY_KEY = 'policy_key'
ALPHA_RANK_PROB = 'alpha_rank_prob'
STEPS_TRAINED = 'steps_trained'
TIME_CREATED = 'time_created'
GAMES_RANKED_ON = 'games_ranked_on'
WALL_TIME_SINCE_START = 'wall_time_since_start'
IS_POLICY_MIXTURE = 'is_policy_mixture'
LBX_SCORE = 'lbx_score'
TAG = 'tag'

RECORD_CSV_FIELDS = [WALL_TIME_SINCE_START, ALPHA_RANK_PROB, POLICY_KEY, STEPS_TRAINED, TIME_CREATED, GAMES_RANKED_ON,
                     IS_POLICY_MIXTURE, LBX_SCORE]

DEFAULT_RECORD_ENTRY_INTERVAL_SECONDS = 0

@ray.remote(num_cpus=1)
class AlphaRankPolicyCatalog:

    def __init__(self,
                 cache_size=0,
                 record_file_path=None,
                 new_record_entry_every_n_seconds=DEFAULT_RECORD_ENTRY_INTERVAL_SECONDS,
                 extra_data_keys=None,
                 store_weights=True,
                 alpha_rank_submission_accept_threshold=None,
                 also_compute_nash_eq=False,
                 track_meta_nash_history_payoff_table=False,
                 save_alpha_rank_html_graph=True,
                 lbx_only_rank_tags: list=None):

        print("\nStarting AlphaRankPolicyCatalog with catch size of {}\n".format(cache_size))

        self.save_alpha_rank_html_graph = save_alpha_rank_html_graph

        self.catalog = OrderedDict()
        self.storage_client = connect_storage_client()
        self.bucket_name = BUCKET_NAME

        self.cache = OrderedDict()
        self.cache_size = cache_size

        self.record_file_path = record_file_path
        self.new_record_entry_every_n_seconds = new_record_entry_every_n_seconds
        self.start_time = time.time()
        self.last_record_entry_time = self.start_time
        self.extra_data_keys = extra_data_keys or []

        self._payoff_table = np.zeros(shape=(0, 0), dtype=np.float32)
        self._alpha_rank_scores = {}

        self.store_weights = store_weights
        self.alpha_rank_submission_accept_threshold = alpha_rank_submission_accept_threshold

        self.lbx_only_rank_tags = lbx_only_rank_tags

        if also_compute_nash_eq:
            import nashpy
        self.also_compute_nash_eq = also_compute_nash_eq

        if track_meta_nash_history_payoff_table:
            self._meta_nash_history_iter = 0
            self._meta_nash_history_catalog = OrderedDict()
            self._meta_nash_history_payoff_table = np.zeros(shape=(0, 0), dtype=np.float32)
            self._meta_nash_history_alpha_rank_scores = {}

    def shutdown(self):
        ray.actor.exit_actor()

    def add_to_cache(self, policy_key, weights):
        assert self.store_weights

        if self.cache_size > 0:
            if policy_key in self.cache:
                self.cache[policy_key] = weights
            else:
                while len(self.cache) >= self.cache_size:
                    # pop oldest item from cache
                    self.cache.popitem(last=False)
                self.cache[policy_key] = weights

    def get_from_cache(self, policy_key):
        assert self.store_weights

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

    def _compute_new_payoff_table(self, old_payoff_table, new_policy_payoff_dict, new_catalog):
        old_size = len(old_payoff_table)
        new_size = old_size + 1

        new_payoff_table = np.zeros(shape=(new_size, new_size), dtype=np.float32)

        if new_size == 1:
            new_payoff_table[0, 0] = 0.0
        else:
            new_payoff_table_row = np.asarray([new_policy_payoff_dict[k] for k in list(new_catalog.keys())[:-1]])
            new_payoff_table[:old_size, :old_size] = old_payoff_table
            new_payoff_table[:old_size, old_size] = -new_payoff_table_row
            new_payoff_table[old_size, :old_size] = new_payoff_table_row

            print("new payoff table entries as new policy: {}".format(new_payoff_table_row))

        # printing out pretty payoff table
        print("Payoff table: (Play as row against col)")
        s = [[str(e) for e in row] for row in new_payoff_table]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        ########

        print(new_payoff_table)
        return new_payoff_table

    def _payoff_table_to_dict(self, payoff_table, catalog):
        payoff_dict = OrderedDict()
        for play_as_idx, play_as_key in enumerate(catalog.keys()):
            payoff_dict[play_as_key] = OrderedDict()
            for play_against_idx, play_against_key in enumerate(catalog.keys()):
                payoff_dict[play_as_key][play_against_key] = payoff_table[play_as_idx, play_against_idx]
        return payoff_dict

    def _compute_alpha_rank_scores(self, payoff_table, catalog):

        pi = get_alpha_rank_pi(payoff_table=payoff_table)

        print("\n\n\n\npi: {}\n\n\n\n".format(pi))

        alpha_rank_scores = {}

        for alpha_rank_score, policy_key in zip(pi, catalog.keys()):
            alpha_rank_scores[policy_key] = alpha_rank_score

        return alpha_rank_scores

    def _compute_worst_loss_scores(self, payoff_table, catalog, lbx_only_rank_tags=None):
        worst_scores = np.min(payoff_table, axis=1)
        print("worst scores:", worst_scores)
        integer_ranking_indexes = np.argsort(worst_scores)
        print("integer_ranking_indexes:", integer_ranking_indexes)

        lbx_scores = OrderedDict()
        policy_keys = list(catalog.keys())
        for rank, idx in enumerate(integer_ranking_indexes):
            policy_key = policy_keys[int(idx)]
            if lbx_only_rank_tags and catalog[policy_key][0][TAG] not in lbx_only_rank_tags:
                continue
            if len(worst_scores) == 1:
                lbx_scores[policy_key] = 0.0
            else:
                lbx_scores[policy_key] = rank / (len(worst_scores) - 1)

        print("lbx scores:", list(lbx_scores.get(policy_key, None) for policy_key in catalog.keys()))
        return lbx_scores

    def _set_store_weights(self, store_weights):
        self.store_weights = store_weights

    def submit_new_policy(self, policy_file_key, policy_keys_to_payoff_dict,
                          steps_trained, ranked_games_played, policy_weights=None, extra_data: dict = None, tag=None):

        assert policy_file_key not in self.catalog

        for existing_catalog_key in self.catalog.keys():
            if existing_catalog_key not in policy_keys_to_payoff_dict:
                raise ValueError("policy_keys_to_payoff_dict needs to contain an entry for every key "
                                 "currently in the catalog, missing: {}".format(existing_catalog_key))

        policy_mixture = None
        data = {STEPS_TRAINED: steps_trained,
                TIME_CREATED: time.time() - self.start_time,
                TAG: tag}

        if extra_data is not None:
            data.update(extra_data)

        new_catalog = self.catalog.copy()
        new_catalog[policy_file_key] = (data, ranked_games_played, policy_mixture)

        new_payoff_table = self._compute_new_payoff_table(old_payoff_table=self._payoff_table,
                                                          new_policy_payoff_dict=policy_keys_to_payoff_dict,
                                                          new_catalog=new_catalog)

        new_alpha_rank_scores = self._compute_alpha_rank_scores(payoff_table=new_payoff_table, catalog=new_catalog)

        if self.also_compute_nash_eq:
            self._compute_nash_eq(payoff_table=new_payoff_table)

        if self.alpha_rank_submission_accept_threshold is not None:
            if new_alpha_rank_scores[policy_file_key] < self.alpha_rank_submission_accept_threshold:
                print("Rejected policy into AlphaRank Policy Catalog, Score: {}".format(new_alpha_rank_scores[policy_file_key]))
                return
            else:
                print("Accepted policy into AlphaRank Policy Catalog, Score: {}".format(new_alpha_rank_scores[policy_file_key]))

        self._payoff_table = new_payoff_table
        self.catalog = new_catalog
        self._alpha_rank_scores = new_alpha_rank_scores

        if self.store_weights:

            if policy_weights is None:
                raise ValueError("Weights must be provided if the store_weights attribute is set to true.")

            self.add_to_cache(policy_key=policy_file_key, weights=policy_weights)

            # save policy weights to disk and cloud
            disk_path = get_default_path_on_disk_for_minio_key(object_name=policy_file_key)
            ensure_dir(disk_path)
            with open(disk_path, "wb") as dill_file:
                dump(obj=policy_weights, file=dill_file)

            upload_file(self.storage_client, bucket_name=self.bucket_name, object_key=policy_file_key, local_source_path=disk_path)

        self.add_record_entry_if_appropriate()

    # def submit_policy_mixture(self, mixture_key, policy_file_keys_to_probs_dict, policy_keys_to_payoff_dict,
    #                           steps_trained, ranked_games_played, extra_data: dict = None):
    #
    #     assert np.isclose(1.0, sum(prob for prob in policy_file_keys_to_probs_dict.values()))
    #     for policy_key in policy_file_keys_to_probs_dict.keys():
    #         assert policy_key in self.catalog
    #
    #     if self.store_weights:
    #         for existing_catalog_key in self.catalog.keys():
    #             if existing_catalog_key not in policy_keys_to_payoff_dict:
    #                 raise ValueError("policy_keys_to_payoff_dict needs to contain an entry for every key "
    #                                  "currently in the catalog")
    #
    #     assert mixture_key not in self.catalog
    #
    #     policy_mixture = policy_file_keys_to_probs_dict
    #     data = {STEPS_TRAINED: steps_trained,
    #             TIME_CREATED: time.time() - self.start_time}
    #
    #     if extra_data is not None:
    #         data.update(extra_data)
    #
    #     new_catalog = self.catalog.copy()
    #     new_catalog[mixture_key] = (data, ranked_games_played, policy_mixture)
    #
    #     new_payoff_table, new_alpha_rank_scores = self._compute_alpha_rank_scores(
    #         old_payoff_table=self._payoff_table,
    #         new_policy_payoff_dict=policy_keys_to_payoff_dict,
    #         new_catalog=new_catalog)
    #
    #     if self.alpha_rank_submission_accept_threshold is not None:
    #         if new_alpha_rank_scores[mixture_key] < self.alpha_rank_submission_accept_threshold:
    #             return
    #
    #     self._payoff_table = new_payoff_table
    #     self.catalog = new_catalog
    #     self._alpha_rank_scores = new_alpha_rank_scores
    #
    #     self.add_record_entry_if_appropriate()

    # def submit_meta_nash_history_payoff_table_columns(self, new_meta_nash_dict, meta_nash_keys_to_payoff_dict):
    #
    #     for key in self._meta_nash_history_catalog:
    #         assert key in meta_nash_keys_to_payoff_dict
    #
    #     self._meta_nash_history_iter += 1
    #
    #     new_meta_nash_catalog = self._meta_nash_history_catalog.copy()
    #     new_meta_nash_catalog[str(self._meta_nash_history_iter)] = new_meta_nash_dict
    #
    #     new_payoff_table, new_alpha_rank_scores = self._compute_alpha_rank_scores(
    #         old_payoff_table=self._meta_nash_history_payoff_table,
    #         new_policy_payoff_dict=meta_nash_keys_to_payoff_dict,
    #         new_catalog=new_meta_nash_catalog)
    #
    #     self._meta_nash_history_catalog = new_meta_nash_catalog
    #     self._meta_nash_history_payoff_table = new_payoff_table
    #     self._meta_nash_history_alpha_rank_scores = new_alpha_rank_scores
    #
    #     self.add_record_entry_if_appropriate()

    def get_random_policy_file_key(self, exclude_keys=None):
        tmp_catalog = self._get_catalog_subset(exclude_keys=exclude_keys)
        selected_key = random.sample(tmp_catalog.keys(), k=1)[0]
        selected_key = self.sample_policy_key_if_key_is_for_mixture(selected_key)
        return selected_key

    def get_policy_key_weighted_by_alpha_rank(self, exploration_coeff=0.0):

        alpha_rank_probs = np.asarray(list(self._alpha_rank_scores.values()))
        uniform = np.ones_like(alpha_rank_probs)/len(alpha_rank_probs)

        selection_probs = exploration_coeff * uniform + (1.0 - exploration_coeff) * alpha_rank_probs

        assert np.isclose(sum(selection_probs), 1.0)

        selected_key = np.random.choice(list(self._alpha_rank_scores.keys()), p=selection_probs)
        return selected_key

    def sample_policy_key_if_key_is_for_mixture(self, policy_key):
        policy_skill_rating, data, games_ranked_on, policy_mixture = self.catalog[policy_key]
        if policy_mixture is None:
            return policy_key
        else:
            return self.sample_policy_key_if_key_is_for_mixture(
                policy_key=np.random.choice(policy_mixture.keys(), p=policy_mixture.values()))

    @ray.method(num_return_vals=2)
    def get_random_policy_weights(self, exclude_keys=None):
        assert self.store_weights

        new_policy_file_key = self.get_random_policy_file_key(exclude_keys=exclude_keys)
        weights = self.get_weights_by_key(new_policy_file_key)
        return weights, new_policy_file_key

    @ray.method(num_return_vals=2)
    def sample_weights_by_alpha_rank(self, exploration_coeff=0.0):
        assert self.store_weights

        new_policy_file_key = self.get_policy_key_weighted_by_alpha_rank(exploration_coeff=exploration_coeff)
        weights = self.get_weights_by_key(new_policy_file_key)
        return weights, new_policy_file_key

    def get_weights_by_key(self, policy_key):
        assert self.store_weights

        weights = self.get_from_cache(policy_key=policy_key)

        if weights is None:
            load_file_path, _ = maybe_download_object(storage_client=self.storage_client, bucket_name=self.bucket_name,
                                                      object_name=policy_key)

            with open(load_file_path, "rb") as dill_file:
                weights = load(file=dill_file)

        return weights

    def get_skill_ranking(self, policy_file_key):
        return self._alpha_rank_scores[policy_file_key]

    def add_record_entry_if_appropriate(self):
        now = time.time()
        if now - self.last_record_entry_time >= self.new_record_entry_every_n_seconds or self.new_record_entry_every_n_seconds < 0:
            self._add_record_entry()
            self.last_record_entry_time = now

    def _add_record_entry(self):
        if self.record_file_path:

            # also save lower bound exploitability ranking
            lbx_scores = self._compute_worst_loss_scores(payoff_table=self._payoff_table,
                                                        catalog=self.catalog,
                                                        lbx_only_rank_tags=self.lbx_only_rank_tags)

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
                    data, games_ranked_on, policy_mixture = vals

                    row = {
                        WALL_TIME_SINCE_START: wall_time,
                        POLICY_KEY: policy_key,
                        ALPHA_RANK_PROB: self._alpha_rank_scores[policy_key],
                        STEPS_TRAINED: data[STEPS_TRAINED],
                        TIME_CREATED: data[TIME_CREATED],
                        GAMES_RANKED_ON: games_ranked_on,
                        IS_POLICY_MIXTURE: policy_mixture is not None,
                        LBX_SCORE: lbx_scores[policy_key] if policy_key in lbx_scores else None}

                    row.update({k: data[k] for k in self.extra_data_keys})

                    writer.writerow(row)

            # save payoff table too
            save_dir = os.path.dirname(self.record_file_path)
            with open(os.path.join(save_dir, 'payoff_table.pkl'), 'wb') as payoff_table_file:
                dump(obj=self._payoff_table, file=payoff_table_file)

            if self.save_alpha_rank_html_graph:
                data = pd.read_csv(self.record_file_path)
                fig = px.line_3d(data, x=TIME_CREATED, y=WALL_TIME_SINCE_START, z=ALPHA_RANK_PROB,
                                 color=WALL_TIME_SINCE_START, color_discrete_sequence=px.colors.sequential.Plotly3,
                                 hover_data=[POLICY_KEY, GAMES_RANKED_ON, STEPS_TRAINED, ALPHA_RANK_PROB, LBX_SCORE, *(key for key in self.extra_data_keys)])
                fig.update_layout(width=1400, height=900, autosize=True)
                fig.layout.scene.xaxis.autorange = 'reversed'
                file_prefix, file_ext = os.path.splitext(self.record_file_path)
                plotly.offline.plot(fig, filename=file_prefix+'.html', auto_open=False)


            # lbx graph
            data = pd.read_csv(self.record_file_path)
            fig = px.line_3d(data.loc[data[LBX_SCORE].notnull()], x=TIME_CREATED, y=WALL_TIME_SINCE_START, z=LBX_SCORE,
                             color=WALL_TIME_SINCE_START, color_discrete_sequence=px.colors.sequential.Plotly3,
                             hover_data=[POLICY_KEY, GAMES_RANKED_ON, STEPS_TRAINED, ALPHA_RANK_PROB, LBX_SCORE,
                                         *(key for key in self.extra_data_keys)])
            fig.update_layout(width=1400, height=900, autosize=True)
            fig.layout.scene.xaxis.autorange = 'reversed'
            file_prefix, file_ext = os.path.splitext(self.record_file_path)
            plotly.offline.plot(fig, filename=file_prefix + '_lbx.html', auto_open=False)

    def get_all_keys(self):
        return list(self.catalog.keys())

    def get_alpha_rank_scores(self):
        return self._alpha_rank_scores.copy()

    def get_current_effective_population_diversity(self):
        return get_effective_population_diversity(
            payoff_table=self._payoff_table,
            pi=np.asarray(list(self._alpha_rank_scores.values())))
