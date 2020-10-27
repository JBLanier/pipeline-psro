import ray
from mprl.utility_services.cloud_storage import connect_storage_client
from mprl.utility_services.worker import ConsoleManagerInterface
from mprl.utility_services.lock_server.lock_client_interface import LockServerInterface
from mprl.utility_services.payoff_table import PayoffTable, PolicySpec
import logging
import time
import dill
logger = logging.getLogger(__name__)
import copy
from termcolor import colored

def _check_consecutive_numbers(int_list, should_start_at=None):
    if len(int_list) == 0:
        return True
    if should_start_at is not None and int_list[0] != should_start_at:
        return False
    prev_num = int_list[0]
    for elem in int_list[1:]:
        if elem != prev_num + 1:
            return False
        prev_num = elem
    return True


def _check_only_latest_policies_are_active(policies_active_states):
    should_all_be_active_now = False
    for is_active in policies_active_states:
        if is_active and not should_all_be_active_now:
            should_all_be_active_now = True
        if should_all_be_active_now and not is_active:
            return False
    return True


@ray.remote(num_cpus=0)
class LivePolicyPayoffTracker(object):
    def __init__(self,
                 minio_bucket,
                 manager_host,
                 manager_port,
                 lock_server_host,
                 lock_server_port,
                 worker_id,
                 policy_class_name,
                 policy_config_key,
                 provide_payoff_barrier_sync=False):
        worker_id = f"live_pop_tracker_{worker_id[worker_id.find('pid'):]}"
        self._storage_client = connect_storage_client()
        self._minio_bucket = minio_bucket
        self._manager_interface = ConsoleManagerInterface(server_host=manager_host,
                                                          port=manager_port,
                                                          worker_id=worker_id,
                                                          storage_client=self._storage_client,
                                                          minio_bucket_name=self._minio_bucket)
        self._lock_interface = LockServerInterface(server_host=lock_server_host,
                                                   port=lock_server_port,
                                                   worker_id=worker_id)
        self._policy_class_name = policy_class_name
        self._policy_config_key = policy_config_key
        self._claimed_policy_num = None
        self._claim_new_active_policy()
        assert self._claimed_policy_num is not None

        self._locally_cached_matchup_results = {}

        self._provide_payoff_barrier_sync = provide_payoff_barrier_sync
        if self._provide_payoff_barrier_sync:
            self._wait_at_payoff_table_barrier_fn, self._leave_barrier_group_fn = self._lock_interface.join_barrier_group(
                barrier_name="pt_barrier",
                member_name=str(self._claimed_policy_num),
                grace_period_for_others_to_join_s=20.0)
        else:
            self._wait_at_payoff_table_barrier_fn = None
            self._leave_barrier_group_fn = None

    @ray.method(num_return_vals=1)
    def wait_at_barrier_for_other_learners(self):
        assert self._provide_payoff_barrier_sync
        self._wait_at_payoff_table_barrier_fn()
        return True

    @ray.method(num_return_vals=1)
    def set_latest_key_for_claimed_policy(self, new_key, request_locks_checkpoint_with_name=None):
        assert self._claimed_policy_num is not None
        prefix = f"policy {self._claimed_policy_num} latest key: "
        new_lock = prefix + new_key
        policy_key_locks = self._lock_interface.get_all_items(filter_by_string=prefix)
        if len(policy_key_locks) > 0:
            assert len(policy_key_locks) == 1
            old_lock = policy_key_locks[0]
            assert self._lock_interface.replace_item(old_item=old_lock,
                                              new_item=new_lock,
                                              new_item_remains_after_disconnect=True,
                                              request_locks_checkpoint_with_name=request_locks_checkpoint_with_name)
            print(colored(f"Policy {self._claimed_policy_num}: Set new latest key for claimed policy (replaced old one): \"{new_lock}\"", 'green'))
        else:
            assert self._lock_interface.try_to_reserve_item(item_name=new_lock,
                                                            remain_after_worker_disconnect=True,
                                                            request_locks_checkpoint_with_name=request_locks_checkpoint_with_name)
            print(colored(f"Policy {self._claimed_policy_num}: Set new latest key for claimed policy: \"{new_lock}\"", "green"))

        return True

    @ray.method(num_return_vals=1)
    def set_claimed_policy_as_finished(self):
        old_lock = f"policy_status: {self._claimed_policy_num} active"
        new_lock = f"policy_status: {self._claimed_policy_num} finished"
        assert self._lock_interface.replace_item(old_item=old_lock,
                                                 new_item=new_lock,
                                                 new_item_remains_after_disconnect=True)
        print(colored(f"Policy {self._claimed_policy_num}: Set claimed policy as finished: \"{new_lock}\"", "green"))
        if self._leave_barrier_group_fn is not None:
            self._leave_barrier_group_fn()
        return True

    @ray.method(num_return_vals=2)
    def get_live_payoff_table_dill_pickled(self, first_wait_for_n_seconds=None):
        if first_wait_for_n_seconds is not None:
            time.sleep(first_wait_for_n_seconds)

        base_payoff_table, _ = self._manager_interface.get_latest_payoff_table(infinite_retry_on_error=False)
        if base_payoff_table is None:
            base_payoff_table = PayoffTable()
        base_payoff_table: PayoffTable = base_payoff_table
        active_policy_numbers, finished_policy_numbers, total_policy_numbers = self.get_active_and_finished_policy_numbers()
        assert len(active_policy_numbers) + len(finished_policy_numbers) == total_policy_numbers
        are_all_lower_policies_finished = len(active_policy_numbers) == 0

        print(colored(f"Policy {self._claimed_policy_num}: There are {total_policy_numbers} policies below this learner. "
                      f"(Active policies below {self._claimed_policy_num} are {active_policy_numbers}. "
                      f"Frozen policies below {self._claimed_policy_num} are {finished_policy_numbers}).", "white"))

        if total_policy_numbers == 0:
            return None, are_all_lower_policies_finished

        assert base_payoff_table.size() <= len(finished_policy_numbers) or base_payoff_table.size() == 1
        missing_policy_nums = list(range(base_payoff_table.size(), total_policy_numbers))
        for missing_policy_num in missing_policy_nums:
            missing_key = self._get_latest_key_for_policy_number(policy_num=missing_policy_num)

            if missing_key is None:
                time.sleep(5)
                missing_key = self._get_latest_key_for_policy_number(policy_num=missing_policy_num)

            if missing_key is not None:
                base_payoff_table.add_policy(new_policy_key=missing_key,
                                             new_policy_class_name=self._policy_class_name,
                                             new_policy_config_file_key=self._policy_config_key,
                                             new_policy_tags=['locally_tracked'])

        required_evals_observed = set()
        required_evals_finalized = set()
        while True:
            matchup_order = base_payoff_table.get_eval_matchup_order()

            if matchup_order is None:
                break
            if matchup_order not in required_evals_finalized:
                as_policy_key, against_policy_key = matchup_order
                payoff, games_played = self._check_eval_cache(as_policy_key=as_policy_key, against_policy_key=against_policy_key)
                if payoff is None:

                    payoff, games_played = self._manager_interface.request_eval_result(
                        as_policy_key=as_policy_key,
                        as_policy_config_key=self._policy_config_key,
                        as_policy_class_name=self._policy_class_name,
                        against_policy_key=against_policy_key,
                        against_policy_config_key=self._policy_config_key,
                        against_policy_class_name=self._policy_class_name,
                        perform_eval_if_not_cached=matchup_order not in required_evals_observed,
                        infinite_retry_on_error=False)

                    if payoff is not None and matchup_order not in required_evals_observed:
                        print(f"{colored(f'Policy {self._claimed_policy_num}: !!!! GOT A CACHE HIT FROM THE MANAGER !!!!','yellow')}\n"
                              f"{colored(f'for {as_policy_key} vs {against_policy_key}', 'yellow')}")

                    if payoff is None and matchup_order in required_evals_observed:
                        print(colored(f"Policy {self._claimed_policy_num}: Waiting to get eval result for {as_policy_key} vs {against_policy_key}", "yellow"))
                        time.sleep(2)

                if payoff is not None:
                    self._add_to_eval_cache_if_not_already_entered(as_policy_key=as_policy_key, against_policy_key=against_policy_key,
                                                                   payoff=payoff, games_played=games_played)
                    base_payoff_table.add_eval_result(as_policy_key=as_policy_key,
                                                      against_policy_key=against_policy_key,
                                                      payoff=payoff,
                                                      games_played=games_played)
                    required_evals_finalized.add(matchup_order)

                required_evals_observed.add(matchup_order)

        assert len(required_evals_observed) >= len(required_evals_finalized)
        assert base_payoff_table.get_num_pending_policies() == 0, f"amount is {base_payoff_table.get_num_pending_policies()}"
        assert base_payoff_table.size() == total_policy_numbers
        return base_payoff_table.to_dill(), are_all_lower_policies_finished

    @ray.method(num_return_vals=1)
    def are_all_lower_policies_finished(self):
        active_policy_numbers, finished_policy_numbers, total_policy_numbers = self.get_active_and_finished_policy_numbers()
        assert len(active_policy_numbers) + len(finished_policy_numbers) == total_policy_numbers
        return len(active_policy_numbers) == 0

    @ray.method(num_return_vals=1)
    def get_claimed_policy_num(self):
        return self._claimed_policy_num

    def get_active_and_finished_policy_numbers(self):
        start_time = time.time()
        while True:
            policy_status_locks = self._lock_interface.get_all_items(filter_by_string="policy_status: ")
            if len(policy_status_locks) == 0:
                return [], [], 0

            _, all_policy_numbers, all_policy_statuses = map(list, zip(*[item.split(" ") for item in policy_status_locks]))
            assert all(stat == "active" or stat == "finished" for stat in all_policy_statuses)

            num_policies_to_consider = self._claimed_policy_num if self._claimed_policy_num is not None else len(all_policy_numbers)
            policy_numbers = [None] * num_policies_to_consider
            policies_active_states = [None] * num_policies_to_consider
            for policy_num, policy_status in zip(all_policy_numbers, all_policy_statuses):
                policy_num = int(policy_num)
                if self._claimed_policy_num is None or policy_num < self._claimed_policy_num:
                    policy_numbers[policy_num] = policy_num
                    policies_active_states[policy_num] = (policy_status == "active")

            if not all(p is not None for p in policy_numbers):
                if time.time() - start_time > 60:
                    raise ValueError(colored(f"policy_numbers (some are None): {policy_numbers}", "red"))
                print(colored(f"policy_numbers (some are None), trying again: {policy_numbers}", "red"))
                time.sleep(0.5)
                continue

            assert all(p is not None for p in policies_active_states)
            assert _check_consecutive_numbers(int_list=policy_numbers, should_start_at=0), f"policy_numbers is {policy_numbers}, all policy status locks are {policy_status_locks}"
            assert _check_only_latest_policies_are_active(policies_active_states=policies_active_states)
            break

        active_policy_numbers = []
        finished_policy_numbers = []
        for i, policy_number in enumerate(policy_numbers):
            if policies_active_states[i]:
                active_policy_numbers.append(policy_number)
            else:
                finished_policy_numbers.append(policy_number)
        total_policy_numbers = len(policy_numbers)
        return active_policy_numbers, finished_policy_numbers, total_policy_numbers

    def _claim_new_active_policy(self):
        if self._claimed_policy_num is not None:
            raise ValueError(f"This interface has already claimed policy {self._claimed_policy_num}")

        _, _, total_policy_numbers = self.get_active_and_finished_policy_numbers()

        claimed_policy_key = self._lock_interface.try_to_reserve_item_from_list(
            possible_item_names_in_order_of_highest_priority_first=[f"policy_status: {i} active" for i in range(total_policy_numbers, total_policy_numbers+100)])
        claimed_policy_num = int(claimed_policy_key.replace('policy_status: ','').replace(' active',''))
        assert claimed_policy_num is not None
        print(colored(f"Claimed Policy {claimed_policy_num}", "green"))
        self._claimed_policy_num = claimed_policy_num
        return claimed_policy_num


    def _get_latest_key_for_policy_number(self, policy_num):
        prefix = f"policy {policy_num} latest key: "
        policy_key_locks = self._lock_interface.get_all_items(filter_by_string=prefix)
        if len(policy_key_locks) == 0:
            return None
        assert len(policy_key_locks) == 1
        policy_key = policy_key_locks[0][len(prefix):]
        return policy_key

    def _check_eval_cache(self, as_policy_key, against_policy_key):
        payoff, games_played = None, None
        try:
            payoff, games_played = self._locally_cached_matchup_results[as_policy_key][against_policy_key]
            print(colored(f"Eval Cache Hit for \"{as_policy_key}\" vs \"{against_policy_key}\"", "green"))
        except KeyError:
            try:
                payoff, games_played = self._locally_cached_matchup_results[against_policy_key][as_policy_key]
                payoff = -payoff
                print(colored(f"Eval Cache Hit for \"{against_policy_key}\" vs \"{as_policy_key}\"", "green"))
            except KeyError:
                pass

        return payoff, games_played

    def _add_to_eval_cache_if_not_already_entered(self, as_policy_key, against_policy_key, payoff, games_played):
        old_payoff, _ = self._check_eval_cache(as_policy_key=as_policy_key, against_policy_key=against_policy_key)
        if old_payoff is not None:
            return

        if as_policy_key not in self._locally_cached_matchup_results:
            self._locally_cached_matchup_results[as_policy_key] = {}
        if against_policy_key not in self._locally_cached_matchup_results[as_policy_key]:
            self._locally_cached_matchup_results[as_policy_key][against_policy_key] = (payoff, games_played)
