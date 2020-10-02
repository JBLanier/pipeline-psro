# %%

# (launch a manager with a fresh payoff table and then run this)


from mprl.utility_services.worker.learner import LearnerManagerInterface
from mprl.utility_services.cloud_storage import connect_storage_client, maybe_download_object
from mprl.utility_services.payoff_table import PayoffTable, PolicySpec
import time
import os

if __name__ == '__main__':


    storage_client = connect_storage_client()

    manager_host = "localhost"
    manager_port = 2828

    new_manager_interface = LearnerManagerInterface(server_host=manager_host,
                                                    port=manager_port,
                                                    worker_id="rebuild_payoff_learner",
                                                    storage_client=storage_client,
                                                    minio_bucket_name="stratego")

    old_payoff_table_local_path, _ = maybe_download_object(storage_client=storage_client,
                                                           bucket_name="stratego",
                                                           object_name="population_server/sage_pid_31932_06_48_20PM_Apr-24-2020/payoff_tables/payoff_table_13_polices_1_pending_sage_pid_31932_07_35_09PM_Apr-25-2020.dill")

    old_payoff_table = PayoffTable.from_dill_file(old_payoff_table_local_path)

    if input(f"You're about to add a bunch of policies to the manager at {manager_host}:{manager_port}\n"
             f"Are you sure? Type \'y\' to go through with this: ") != 'y':
        print("(doing nothing and exiting)")
        exit(0)


    for index in range(old_payoff_table.size()):
        policy: PolicySpec = old_payoff_table.get_policy_for_index(index=index)

        new_manager_interface.submit_new_policy_for_population(policy_weights_key=policy.key,
                                                               policy_config_key=policy.config_key,
                                                               policy_class_name=policy.class_name,
                                                               policy_tags=policy.tags,
                                                               infinite_retry_on_error=False)
        if index > 0:
            time.sleep(60 * 60)

    for policy in old_payoff_table.get_pending_policy_specs():
        new_manager_interface.submit_new_policy_for_population(policy_weights_key=policy.key,
                                                               policy_config_key=policy.config_key,
                                                               policy_class_name=policy.class_name,
                                                               policy_tags=policy.tags,
                                                               infinite_retry_on_error=False)

