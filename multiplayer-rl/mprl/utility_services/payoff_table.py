import logging
from queue import Queue, Empty

import dill
import numpy as np
from bidict import bidict
from copy import deepcopy
logger = logging.getLogger(__name__)


class PolicySpec(object):

    def __init__(self, policy_key, policy_class_name, policy_config_key, tags=None, payoff_matrix_index=None):
        self.key = policy_key
        self.class_name = policy_class_name
        self.config_key = policy_config_key
        self.tags = [] if tags is None else tags
        self._payoff_matrix_index = payoff_matrix_index

    def assign_payoff_matrix_index(self, index):
        assert self._payoff_matrix_index is None
        logger.debug(f"Assigned matrix index {index} to policy {self.key}")
        self._payoff_matrix_index = index

    def get_payoff_matrix_index(self):
        return self._payoff_matrix_index


class PayoffTable(object):

    class _P2SROPLegacyUnpickler(dill.Unpickler):
        # Accounts for a change in module name since the time certain old payoff tables were pickled.
        def find_class(self, module, name):
            if module == 'population_server.payoff_table':
                module = 'mprl.utility_services.payoff_table'
            return super().find_class(module, name)

    @classmethod
    def from_dill_file(cls, dill_file_path):
        with open(dill_file_path, 'rb') as dill_file:
            payoff_table = cls._P2SROPLegacyUnpickler(dill_file).load()
        assert isinstance(payoff_table,
                          PayoffTable), f"The loaded item at {dill_file_path} was not a PayoffTable instance."
        return payoff_table

    def __init__(self):
        self._matrix_index_to_policies = bidict()
        self._payoff_matrix_as_rows_against_columns = np.asarray([[]], np.float32)
        self._games_played_matrix = np.asarray([[]], np.int32)

        self._policy_keys_to_matrix_index = {policy.key: index for index, policy in
                                             self._matrix_index_to_policies.items()}

        # Keep track of eval results for new policies until we have results against all existing policies in table.
        # Once we have all matchups fulfilled, we'll add these results to the payoff table proper.
        self._pending_eval_matchups = Queue()
        self._fulfilled_eval_matchups_pending_insertion_to_table = {}
        self._pending_policies_by_key = {}

    def save_to_dill_file_(self, save_file_path):
        logger.debug(f"Saving payoff table to {save_file_path}")
        with open(save_file_path, '+wb') as dill_file:
            dill.dump(self, dill_file)

    def to_dill(self):
        return dill.dumps(self)

    def size(self):
        return len(self._matrix_index_to_policies)

    def get_num_pending_policies(self):
        return len(self._pending_policies_by_key)

    def get_filtered_payoff_matrix(self,
                                   accepted_policy_class_names=None,
                                   accepted_model_config_keys=None,
                                   add_payoff_matrix_noise_std_dev=None,
                                   only_first_n_policies=None,
                                   leave_out_indexes=None):

        filtered_payoff_matrix = self._payoff_matrix_as_rows_against_columns.copy()

        # ensure that matrix is anti-symmetric
        assert (filtered_payoff_matrix.transpose() == -filtered_payoff_matrix).all()

        if add_payoff_matrix_noise_std_dev:
            for i in range(len(filtered_payoff_matrix)):
                for j in range(i):
                    noise_amt = np.random.normal(loc=0, scale=add_payoff_matrix_noise_std_dev)
                    filtered_payoff_matrix[i, j] += noise_amt
                    filtered_payoff_matrix[j, i] -= noise_amt

        # ensure that matrix is anti-symmetric
        assert (filtered_payoff_matrix.transpose() == -filtered_payoff_matrix).all()

        # returns a payoff matrix where 'non-accepted' policies have the worst possible payoff stats
        # so a metanash solver should set their selection probs to zero.
        # assumes zero-sum game payoff table (1 for always win, -1 for always lose)
        if leave_out_indexes is not None:
            filter_indexes = leave_out_indexes
        else:
            filter_indexes = []
        for i, policy in self._matrix_index_to_policies.items():
            if accepted_policy_class_names is not None:
                if policy.class_name not in accepted_policy_class_names:
                    filter_indexes.append(i)
                    continue
            if accepted_model_config_keys is not None:
                if policy.config_key not in accepted_model_config_keys:
                    filter_indexes.append(i)
                    continue
        for filter_index in filter_indexes:
            filtered_payoff_matrix[:, filter_index] = 1.0
        for filter_index in filter_indexes:
            filtered_payoff_matrix[filter_index, :] = -1.0
        for filter_index in filter_indexes:
            filtered_payoff_matrix[filter_index, filter_index] = 0.0

        if only_first_n_policies is not None:
            filtered_payoff_matrix = filtered_payoff_matrix[:only_first_n_policies, :only_first_n_policies]

        return filtered_payoff_matrix

    def get_payoff_matrix(self):
        return self._payoff_matrix_as_rows_against_columns.copy()

    def get_policy_for_index(self, index):
        return self._matrix_index_to_policies[index]

    def is_policy_key_in_payoff_matrix(self, policy_key):
        try:
            _ = self._policy_keys_to_matrix_index[policy_key]
            return True
        except KeyError:
            return False

    def get_policy_spec_for_key(self, key):
        try:
            index = self._policy_keys_to_matrix_index[key]
            return self._matrix_index_to_policies[index]
        except KeyError:
            return self._pending_policies_by_key[key]

    def get_pending_policy_specs(self):
        return list(self._pending_policies_by_key.values())

    def get_ordered_keys_in_payoff_matrix(self):
        return [self._matrix_index_to_policies[i].key for i in range(len(self._matrix_index_to_policies))]

    def add_policy(self, new_policy_key, new_policy_class_name, new_policy_config_file_key, new_policy_tags):
        logger.info(f"Adding new policy to payoff table (pending evals): {new_policy_key}")
        new_policy = PolicySpec(policy_key=new_policy_key,
                                policy_class_name=new_policy_class_name,
                                policy_config_key=new_policy_config_file_key,
                                tags=list.copy(list(new_policy_tags)),
                                payoff_matrix_index=None)

        if len(self._payoff_matrix_as_rows_against_columns) == 0:
            self._add_policy_to_payoff_table(policy=new_policy, against_policies_to_payoffs_and_games_played={})

        assert self._fulfilled_eval_matchups_pending_insertion_to_table.get(new_policy) is None
        assert self._policy_keys_to_matrix_index.get(new_policy) is None

        # add new pending eval matchups
        self._fulfilled_eval_matchups_pending_insertion_to_table[new_policy.key] = {}
        self._add_missing_pending_eval_matchups_for_policy(new_policy.key)
        self._pending_policies_by_key[new_policy.key] = new_policy
        self._add_pending_policy_to_payoff_table_if_all_matchups_are_fulfilled(policy_key=new_policy_key)

    def get_eval_matchup_order(self):
        try:
            while True:
                as_policy_key, against_policy_key = self._pending_eval_matchups.get_nowait()
                if not self._is_policy_matchup_already_fulfilled(as_policy_key=as_policy_key,
                                                                 against_policy_key=against_policy_key):
                    break
        except Empty:
            return None

        # put it in the back of the queue in case this order doesnt actually get fulfilled
        self._pending_eval_matchups.put((as_policy_key, against_policy_key))

        return as_policy_key, against_policy_key

    def _clear_fulfilled_matchup_orders_from_queue(self):
        # TODO this can be implemented better
        orig_queue_length = self._pending_eval_matchups.qsize()
        for _ in range(orig_queue_length):
            try:
                as_policy_key, against_policy_key = self._pending_eval_matchups.get_nowait()
                if not self._is_policy_matchup_already_fulfilled(as_policy_key=as_policy_key,
                                                                 against_policy_key=against_policy_key):
                    # put it back
                    self._pending_eval_matchups.put((as_policy_key, against_policy_key))
            except Empty:
                break

    def add_eval_result(self, as_policy_key, against_policy_key, payoff, games_played):
        if self._fulfilled_eval_matchups_pending_insertion_to_table.get(as_policy_key) is None:
            logger.info(f"Eval result wasn\'t added to payoff table because it was already fulfilled:\n as {as_policy_key}\nvs\n{against_policy_key}\n payoff {payoff}, {games_played} games played")
            return False
        # if self._policy_keys_to_matrix_index.get(against_policy_key) is None:
        #     logger.info(f"Eval result wasn\'t added to payoff table because the 2nd policy "
        #                 f"isn't in the payoff matrix:\n"
        #                 f"as {as_policy_key}\nvs\n{against_policy_key}\n"
        #                 f"payoff {payoff}, {games_played} games played")
        #     return False
        if self._fulfilled_eval_matchups_pending_insertion_to_table[as_policy_key].get(against_policy_key) is not None:
            prev_payoff, prev_games = self._fulfilled_eval_matchups_pending_insertion_to_table[as_policy_key][
                against_policy_key]
            logger.info(f"Eval result wasn\'t added to payoff table because the matchup was already fulfilled:",
                        f"as {as_policy_key}\nvs\n{against_policy_key}\n"
                        f"Submitted: payoff {payoff}, {games_played} games played\n"
                        f"Already recorded: payoff {payoff}, {prev_games} games played")
            return False

        self._fulfilled_eval_matchups_pending_insertion_to_table[as_policy_key][against_policy_key] = (
        payoff, games_played)

        # add reverse if it's also pending
        if self._policy_keys_to_matrix_index.get(against_policy_key) is None:
            if self._fulfilled_eval_matchups_pending_insertion_to_table.get(against_policy_key) is None:
                self._fulfilled_eval_matchups_pending_insertion_to_table[against_policy_key] = {}

            self._fulfilled_eval_matchups_pending_insertion_to_table[against_policy_key][as_policy_key] = (
            -payoff, games_played)


        logger.info(f"Eval result added to payoff table (pending add to payoff matrix):\n"
                    f"as {as_policy_key}\nvs\n{against_policy_key}\n"
                    f"payoff {payoff}, {games_played} games played")

        was_policy_added_to_payoff_table = self._add_pending_policy_to_payoff_table_if_all_matchups_are_fulfilled(
            policy_key=as_policy_key)

        # refresh what matchups still need to be evaluated for all pending policies
        for pending_policy_key in self._pending_policies_by_key.keys():
            self._add_missing_pending_eval_matchups_for_policy(policy_key=pending_policy_key)

        self._clear_fulfilled_matchup_orders_from_queue()

        return was_policy_added_to_payoff_table

    def get_pending_policy_stats(self):
        return {
            'pending_policy_matchups': list(self._pending_eval_matchups.queue),
            'policies_pending_insertion_to_matrix': list(self._pending_policies_by_key.keys())
        }

    def _is_policy_matchup_already_fulfilled(self, as_policy_key, against_policy_key):
        if self._policy_keys_to_matrix_index.get(as_policy_key) is not None:
            # logger.debug(f"{as_policy_key} is already in the matrix")
            return True
        try:
            _ = self._fulfilled_eval_matchups_pending_insertion_to_table[as_policy_key][against_policy_key]
            # logger.debug(f"{as_policy_key}\nvs\n {against_policy_key}\nwas already fulfilled")
            return True
        except KeyError:
            return False

    def _add_missing_pending_eval_matchups_for_policy(self, policy_key):
        matchups_added = False

        # TODO this be implemented better

        for pending_against_policy_key in self._pending_policies_by_key.keys():
            if pending_against_policy_key != policy_key:
                if self._fulfilled_eval_matchups_pending_insertion_to_table[policy_key].get(pending_against_policy_key) is None:
                    assert isinstance(policy_key, str)
                    assert isinstance(pending_against_policy_key, str)
                    self._pending_eval_matchups.put((policy_key, pending_against_policy_key))
                    matchups_added = True

        all_matrix_against_policy_keys = [pol.key for pol in self._matrix_index_to_policies.values()]
        for against_policy_key in all_matrix_against_policy_keys:
            if self._fulfilled_eval_matchups_pending_insertion_to_table[policy_key].get(against_policy_key) is None:
                if (policy_key, against_policy_key) not in self._pending_eval_matchups.queue:
                    assert isinstance(policy_key, str)
                    assert isinstance(against_policy_key, str)
                    self._pending_eval_matchups.put((policy_key, against_policy_key))
                    matchups_added = True
        return matchups_added

    def _add_pending_policy_to_payoff_table_if_all_matchups_are_fulfilled(self, policy_key):
        all_possible_against_policy_keys = list(self._policy_keys_to_matrix_index.keys())
        if self._fulfilled_eval_matchups_pending_insertion_to_table.get(policy_key) is None:
            return False
        for against_policy_key in all_possible_against_policy_keys:
            if self._fulfilled_eval_matchups_pending_insertion_to_table[policy_key].get(against_policy_key) is None:
                return False

        fulfilled_matchups = deepcopy(self._fulfilled_eval_matchups_pending_insertion_to_table[policy_key])
        to_delete = []
        for key in fulfilled_matchups.keys():
            if key not in all_possible_against_policy_keys:
                to_delete.append(key)
        for key in to_delete:
            del fulfilled_matchups[key]

        self._add_policy_to_payoff_table(policy=self._pending_policies_by_key[policy_key],
                                         against_policies_to_payoffs_and_games_played=fulfilled_matchups)

        del self._fulfilled_eval_matchups_pending_insertion_to_table[policy_key]
        del self._pending_policies_by_key[policy_key]

        return True

    def _add_policy_to_payoff_table(self, policy: PolicySpec, against_policies_to_payoffs_and_games_played):
        assert len(self._matrix_index_to_policies) == 0 or \
               np.shape(self._payoff_matrix_as_rows_against_columns)[0] == \
               np.shape(self._payoff_matrix_as_rows_against_columns)[1]
        new_policy_index = len(self._matrix_index_to_policies)

        policy.assign_payoff_matrix_index(index=new_policy_index)
        self._matrix_index_to_policies[new_policy_index] = policy

        if len(self._matrix_index_to_policies) == 1:
            if against_policies_to_payoffs_and_games_played is not None and \
                    len(against_policies_to_payoffs_and_games_played) > 0:
                raise ValueError("No games should be played for the first policy in the payoff table.")
            self._payoff_matrix_as_rows_against_columns = np.asarray([[0.0]], dtype=np.float32)
            self._games_played_matrix = np.asarray([[0]], dtype=np.int32)
        else:
            # expand payoff matrix
            self._payoff_matrix_as_rows_against_columns = np.pad(self._payoff_matrix_as_rows_against_columns,
                                                                 ((0, 1), (0, 1)))
            self._games_played_matrix = np.pad(self._games_played_matrix, ((0, 1), (0, 1)))
            for against_policy_key, (payoff, games_played) in against_policies_to_payoffs_and_games_played.items():
                against_policy_index = self._policy_keys_to_matrix_index[against_policy_key]
                self._payoff_matrix_as_rows_against_columns[new_policy_index, against_policy_index] = payoff
                self._payoff_matrix_as_rows_against_columns[against_policy_index, new_policy_index] = -payoff

                self._games_played_matrix[new_policy_index, against_policy_index] = games_played
                self._games_played_matrix[against_policy_index, new_policy_index] = games_played
        assert policy.get_payoff_matrix_index() is not None
        self._policy_keys_to_matrix_index[policy.key] = policy.get_payoff_matrix_index()
        logger.info(f"Added pending policy to payoff matrix: {policy.key}")
        logger.debug(f"New payoff matrix:\n{self._payoff_matrix_as_rows_against_columns}")
        logger.debug(f"New games_played matrix:\n{self._games_played_matrix}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import dill

    p = PayoffTable(from_file_key=None, storage_client=None, bucket_name=None)
    p.add_policy(new_policy_key="fds",
                 new_policy_class_name="Fd",
                 new_policy_config_file_key="fds",
                 new_policy_tags="fd")
    with open("test_payoff_table.dill", "+wb") as f:
        dill.dump(p, f)
