
import os
from mprl.utility_services.cloud_storage import connect_storage_client
from mprl.utility_services.worker import ConsoleManagerInterface
from mprl.utility_services.lock_server.lock_client_interface import LockServerInterface
from mprl.utility_services.payoff_table import PayoffTable, PolicySpec
import logging
from socket import gethostname
import time
from mprl.scripts.poker_parallel_algos.utils.metanash import get_fp_metanash_for_latest_payoff_table

MANAGER_SERVER_HOST = os.getenv("MANAGER_SERVER_HOST", "localhost")
MANAGER_PORT = int(os.getenv("MANAGER_PORT"))
if not MANAGER_PORT:
    raise ValueError("Environment variable MANAGER_PORT needs to be set.")

LOCK_SERVER_HOST = os.getenv("LOCK_SERVER_HOST", 'localhost')
LOCK_SERVER_PORT = int(os.getenv("LOCK_SERVER_PORT"))
if not LOCK_SERVER_PORT:
    raise ValueError("Environment variable LOCK_SERVER_PORT needs to be set.")

JOB_STATUS_OPEN = 'open'
JOB_STATUS_ACTIVE = 'active'
def _job_str_for_policy_key(policy_key):
    return f"job: {policy_key} {JOB_STATUS_OPEN}"

if __name__ == '__main__':
    storage_client = connect_storage_client()

    manager_interface = ConsoleManagerInterface(server_host=MANAGER_SERVER_HOST,
                                           port=MANAGER_PORT,
                                           worker_id=f"rectified_psro_job_scheduler_{gethostname()}_pid_{os.getpid()}",
                                           storage_client=storage_client,
                                           minio_bucket_name=BUCKET_NAME)

    lock_server_interface = LockServerInterface(server_host=LOCK_SERVER_HOST,
                                                   port=LOCK_SERVER_PORT,
                                                   worker_id=f"rectified_psro_job_scheduler_{gethostname()}_pid_{os.getpid()}")

    generation_index = 0
    while True:
        print(f"Starting generation {generation_index}")
        active_job_list = []
        active_job_policy_keys = []
        if generation_index == 0:
            random_policy_key = "random"
            vs_random_job_str = _job_str_for_policy_key(policy_key=random_policy_key)
            assert lock_server_interface.try_to_reserve_item(item_name=vs_random_job_str)
            active_job_list.append(vs_random_job_str)
            active_job_policy_keys.append(random_policy_key)
        else:
            # get metanash probs and make jobs for non-zero policies
            selection_probs, payoff_table, payoff_table_key = get_fp_metanash_for_latest_payoff_table(manager_interface=manager_interface,
                                                                                    fp_iters=20000,
                                                                                    add_payoff_matrix_noise_std_dev=0.0,
                                                                                    mix_with_uniform_dist_coeff=None)
            for policy_key, prob in zip(payoff_table.get_ordered_keys_in_payoff_matrix(), selection_probs):
                if prob > 0:
                    job_str = _job_str_for_policy_key(policy_key=policy_key)
                    assert lock_server_interface.try_to_reserve_item(item_name=job_str)
                    active_job_list.append(job_str)
                    active_job_policy_keys.append(policy_key)
        print(f"\n\n\nLaunched the following jobs for generation {generation_index}:")
        for job in active_job_list:
            print(f"\t{job}")

        # wait for jobs to be marked as completed
        submitted_job_keys = []
        while True:
            jobs_locks = lock_server_interface.get_all_items(filter_by_string="job: ")
            found_job_policy_keys = set()
            pending_jobs = []
            all_jobs_finished = True
            for job in jobs_locks:
                _, policy_key, status = job.split(' ')
                found_job_policy_keys.add(policy_key)
                if status != JOB_STATUS_OPEN and status != JOB_STATUS_ACTIVE:
                    # completed job status should be replaced with a submitted policy key
                    assert ".dill" in status
                    submitted_job_keys.append(status)
                else:
                    all_jobs_finished = False
                    pending_jobs.append(job)
            assert found_job_policy_keys == set(active_job_policy_keys)
            if not all_jobs_finished:
                print(f"\n\n\nWaiting on {len(pending_jobs)} jobs:")
                for pending_job in pending_jobs:
                    print(f"\t{pending_job}")
                time.sleep(10)
            else:
                break

        # clear job keys
        jobs_locks = lock_server_interface.get_all_items(filter_by_string="job: ")
        for job in jobs_locks:
            assert lock_server_interface.release_item(job)

        # wait for submitted keys to show up in payoff matrix
        while True:
            all_payoffs_done_and_ready_for_next_generation = True
            for submitted_job_policy_key in submitted_job_keys:
                if not manager_interface.is_policy_key_in_current_payoff_matrix(policy_key=submitted_job_policy_key):
                    all_payoffs_done_and_ready_for_next_generation = False
                    print(f"Waiting on policy {submitted_job_policy_key} to show up in payoff matrix")
                    time.sleep(10)
            if all_payoffs_done_and_ready_for_next_generation:
                break

        print(f"Finished generation {generation_index}")
        generation_index += 1

