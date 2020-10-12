import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent import futures
from queue import PriorityQueue, Empty, Queue
from socket import gethostname
from threading import Lock, RLock
from termcolor import colored
import grpc

from mprl.utility_services.cloud_storage import connect_storage_client, upload_file, get_default_path_on_disk_for_minio_key, post_key
from mprl.utility_services.cloud_storage import maybe_download_object, MINIO_ENDPOINT, BUCKET_NAME, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
from mprl.utility_services.payoff_table import PayoffTable, PolicySpec
from mprl.utility_services.protobuf.population_server_pb2 import WorkerPing, PayoffTableKey, PolicyInfo, Confirmation, \
    EvalMatchupOrder, EvalMatchupResult, ManagerStats
from mprl.utility_services.protobuf.population_server_pb2_grpc import PopulationServerServicer, \
    add_PopulationServerServicer_to_server
from mprl.utility_services.utils import datetime_str, ensure_dir, seconds_to_text

logger = logging.getLogger(__name__)

IGNORE_REPEAT_EVAL_REQUESTS_INTERVAL_SECONDS = 60 * 10
CLOUD_PREFIX = os.getenv("CLOUD_PREFIX", "")

class _PopulationServerServicerImpl(PopulationServerServicer):

    def __init__(self,
                 stop_event,
                 payoff_table_save_key_prefix_dir,
                 storage_client,
                 bucket_name,
                 max_ping_interval_seconds_to_track_workers,
                 num_games_to_play_for_matchup_evals,
                 restore_from_payoff_table_key=None):

        self._stop_event = stop_event
        self.payoff_table_save_key_prefix_dir = payoff_table_save_key_prefix_dir
        self._storage_client = storage_client
        self._bucket_name = bucket_name
        self._max_ping_interval_seconds_to_track_workers = max_ping_interval_seconds_to_track_workers
        self._num_games_to_play_for_matchup_evals = num_games_to_play_for_matchup_evals

        self._payoff_table_modification_lock = Lock()
        self._recent_worker_pings = PriorityQueue()
        self._worker_ping_modification_lock = Lock()
        self._start_time = time.time()

        self._eval_matchup_cache_lock = RLock()
        self._eval_matchup_cache = {}
        self._externally_requested_eval_queue = Queue()

        self._recent_eval_match_requests_lock = RLock()
        self._recent_eval_match_requests = {}

        self._latest_checkpoint_key = os.path.join(self.payoff_table_save_key_prefix_dir, "latest.dill")
        logger.info(colored(f"Latest Manager Payoff Table Checkpoint will always be at {self._latest_checkpoint_key} "
                            f"(local file path: {get_default_path_on_disk_for_minio_key(self._latest_checkpoint_key)})", "yellow"))

        if restore_from_payoff_table_key is not None:
            payoff_table_local_path, _ = maybe_download_object(storage_client=self._storage_client,
                                                               bucket_name=self._bucket_name,
                                                               object_name=restore_from_payoff_table_key,
                                                               force_download=False)
            logger.info(f"restoring payoff table from {payoff_table_local_path}")
            self._payoff_table = PayoffTable.from_dill_file(dill_file_path=payoff_table_local_path)
            self._latest_payoff_table_key = restore_from_payoff_table_key
            self._log_policies_in_payoff_matrix()
        else:
            logger.info(f"creating new empty payoff table with no policies")
            self._payoff_table = PayoffTable()
            self._latest_payoff_table_key = None

    def _log_policies_in_payoff_matrix(self):
        out = "Policies in matrix:"
        with self._payoff_table_modification_lock:
            for policy_key in self._payoff_table.get_ordered_keys_in_payoff_matrix():
                out += f"\n{policy_key}"
        out += "\n"
        logger.info(out)

    # Implementations of gRPC functions that other programs (learners, evaluators, etc) will call.
    def _remove_old_worker_pings(self):
        if not self._recent_worker_pings.empty():
            while True:
                try:
                    ping_time, worker_ping = self._recent_worker_pings.get_nowait()
                except Empty:
                    break
                # logger.debug(f"oldest ping is {time.time() - ping_time} seconds old")
                if time.time() - ping_time < self._max_ping_interval_seconds_to_track_workers:
                    # put it back if it's not too old
                    self._recent_worker_pings.put((ping_time, worker_ping))
                    break

    def Ping(self, request: WorkerPing, context):
        # logger.debug(f"Got worker ping for {request.worker_type}, worker id {request.worker_id}")
        with self._worker_ping_modification_lock:
            self._recent_worker_pings.put((time.time(), request))
            self._remove_old_worker_pings()

        response = Confirmation()
        response.confirmation = True
        return response

    def GetManagerStats(self, request, context):
        with self._worker_ping_modification_lock:
            self._remove_old_worker_pings()

        now = time.time()
        manager_stats = OrderedDict()

        worker_stats = {}
        worker_pings_list = list(self._recent_worker_pings.queue)
        for ping_time, worker_ping in worker_pings_list:
            worker_id = worker_ping.worker_id
            worker_type = worker_ping.worker_type
            if worker_type not in worker_stats:
                worker_stats[worker_type] = {}

            time_since_last_ping = now - ping_time
            if worker_id in worker_stats[worker_type]:
                if worker_stats[worker_type][worker_id]["time since last ping"] > time_since_last_ping:
                    worker_stats[worker_type][worker_id]["time since last ping"] = time_since_last_ping
            else:
                worker_stats[worker_type][worker_id] = {"time since last ping": time_since_last_ping}

        for worker_type in worker_stats.keys():
            for worker_id in worker_stats[worker_type].keys():
                worker_stats[worker_type][worker_id][
                    "time since last ping"] = f"{worker_stats[worker_type][worker_id]['time since last ping']:.1f}s"

        manager_stats["Manager Uptime"] = seconds_to_text(now - self._start_time)
        manager_stats["Worker Stats"] = worker_stats

        with self._payoff_table_modification_lock:
            manager_stats["Payoff Matrix Size"] = f"{self._payoff_table.size()} policies"
            manager_stats["Payoff Table Pending Policy Stats"] = self._payoff_table.get_pending_policy_stats()

        all_stats = {"manager_hostname": gethostname(), "stats": manager_stats}
        response = ManagerStats()
        response.manager_stats_json = json.dumps(all_stats)

        return response

    def GetLatestPayoffTableKey(self, request, context):
        latest_payoff_table_key = self._latest_payoff_table_key
        response = PayoffTableKey()
        response.payoff_table_is_empty = latest_payoff_table_key is None
        response.key = "" if latest_payoff_table_key is None else latest_payoff_table_key
        return response

    def SubmitNewPolicyForPopulation(self, request: PolicyInfo, context):
        new_policy_key = request.policy_key
        new_policy_model_config_key = request.policy_model_config_key
        new_policy_class_name = request.policy_class_name
        new_policy_tags = request.policy_tags

        with self._payoff_table_modification_lock:
            self._payoff_table.add_policy(new_policy_key=new_policy_key, new_policy_class_name=new_policy_class_name,
                                          new_policy_config_file_key=new_policy_model_config_key,
                                          new_policy_tags=new_policy_tags)
            self._checkpoint_payoff_table()

        response = Confirmation()
        response.confirmation = True
        return response

    def RequestEvalMatchup(self, request, context):

        try:
            response = self._externally_requested_eval_queue.get_nowait()
        except Empty:
            with self._payoff_table_modification_lock:
                response = EvalMatchupOrder()

                while True:
                    policy_pair = self._payoff_table.get_eval_matchup_order()

                    if policy_pair is None:
                        response.no_matchups_needed = True
                        # response.as_policy = PolicyInfo()
                        # response.against_policy = PolicyInfo()
                        response.num_games_to_play = 0
                        break
                    else:
                        as_policy_key, against_policy_key = policy_pair
                        existing_payoff, existing_games_played = self._check_eval_cache(as_policy_key=as_policy_key,
                                                                                        against_policy_key=against_policy_key)
                        if existing_payoff is not None:
                            new_policy_was_added_to_payoff_table = self._payoff_table.add_eval_result(
                                as_policy_key=as_policy_key,
                                against_policy_key=against_policy_key,
                                payoff=existing_payoff,
                                games_played=existing_games_played)
                            if new_policy_was_added_to_payoff_table:
                                self._checkpoint_payoff_table()
                        else:
                            as_policy_spec: PolicySpec = self._payoff_table.get_policy_spec_for_key(as_policy_key)
                            against_policy_spec: PolicySpec = self._payoff_table.get_policy_spec_for_key(against_policy_key)
                            response.no_matchups_needed = False

                            response.as_policy.policy_key = as_policy_spec.key
                            response.as_policy.policy_model_config_key = as_policy_spec.config_key
                            response.as_policy.policy_class_name = as_policy_spec.class_name
                            response.as_policy.policy_tags.extend(as_policy_spec.tags)

                            response.against_policy.policy_key = against_policy_spec.key
                            response.against_policy.policy_model_config_key = against_policy_spec.config_key
                            response.against_policy.policy_class_name = against_policy_spec.class_name
                            response.against_policy.policy_tags.extend(against_policy_spec.tags)

                            response.num_games_to_play = self._num_games_to_play_for_matchup_evals
                            break
        return response

    def RequestEvalResult(self, request, context):
        matchup_order = request.matchup
        perform_eval_if_not_cached = request.perform_eval_if_not_cached
        as_policy_key = matchup_order.as_policy.policy_key
        against_policy_key = matchup_order.against_policy.policy_key
        payoff, games_played = self._check_eval_cache(as_policy_key=as_policy_key, against_policy_key=against_policy_key)

        if payoff is None and perform_eval_if_not_cached:
            with self._eval_matchup_cache_lock:
                self._flush_old_eval_match_requests()
                if (as_policy_key, against_policy_key) not in self._recent_eval_match_requests and \
                        (against_policy_key, as_policy_key) not in self._recent_eval_match_requests:
                    self._recent_eval_match_requests[(as_policy_key, against_policy_key)] = time.time()
                    matchup_order.num_games_to_play = self._num_games_to_play_for_matchup_evals
                    print(colored(f"External Eval Queued for \"{as_policy_key}\" vs \"{against_policy_key}\"", "yellow"))
                    self._externally_requested_eval_queue.put(matchup_order)

        if payoff is None:
            payoff_response = -1
            games_played_response = -1
        else:
            payoff_response = payoff
            games_played_response = games_played

        assert payoff_response is not None
        assert games_played_response is not None

        response = EvalMatchupResult()
        response.as_policy_key = as_policy_key
        response.against_policy_key = against_policy_key
        response.payoff = payoff_response
        response.games_played = games_played_response
        return response

    def _flush_old_eval_match_requests(self):
        now = time.time()
        to_delete = []
        with self._recent_eval_match_requests_lock:
            for matchup, request_time in self._recent_eval_match_requests.items():
                if now - request_time > IGNORE_REPEAT_EVAL_REQUESTS_INTERVAL_SECONDS:
                    to_delete.append(matchup)
            for matchup in to_delete:
                del self._recent_eval_match_requests[matchup]

    def SubmitEvalMatchupResult(self, request: EvalMatchupResult, context):

        as_policy_key = request.as_policy_key
        against_policy_key = request.against_policy_key
        payoff = request.payoff
        games_played = request.games_played

        self._add_to_eval_cache_if_not_already_entered(
            as_policy_key=as_policy_key,
            against_policy_key=against_policy_key,
            payoff=payoff,
            games_played=games_played
        )

        with self._payoff_table_modification_lock:

            policy_is_irrelevant_to_payoff_table = False
            try:
                self._payoff_table.get_policy_spec_for_key(as_policy_key)
                self._payoff_table.get_policy_spec_for_key(against_policy_key)
            except KeyError:
                policy_is_irrelevant_to_payoff_table = True

            if not policy_is_irrelevant_to_payoff_table:
                new_policy_was_added_to_payoff_table = self._payoff_table.add_eval_result(
                    as_policy_key=as_policy_key,
                    against_policy_key=against_policy_key,
                    payoff=payoff,
                    games_played=games_played)

                if new_policy_was_added_to_payoff_table:
                    self._checkpoint_payoff_table()

        response = Confirmation()
        response.confirmation = True
        return response

    def _check_eval_cache(self, as_policy_key, against_policy_key):
        payoff, games_played = None, None
        with self._eval_matchup_cache_lock:
            try:
                payoff, games_played = self._eval_matchup_cache[as_policy_key][against_policy_key]
                print(colored(f"Eval Cache Hit for \"{as_policy_key}\" vs \"{against_policy_key}\"", "green"))
            except KeyError:
                try:
                    payoff, games_played = self._eval_matchup_cache[against_policy_key][as_policy_key]
                    payoff = -payoff
                    print(colored(f"Eval Cache Hit for \"{against_policy_key}\" vs \"{as_policy_key}\"", "green"))
                except KeyError:
                    pass

        return payoff, games_played

    def _add_to_eval_cache_if_not_already_entered(self, as_policy_key, against_policy_key, payoff, games_played):
        with self._eval_matchup_cache_lock:
            if as_policy_key not in self._eval_matchup_cache:
                self._eval_matchup_cache[as_policy_key] = {}
            if against_policy_key not in self._eval_matchup_cache[as_policy_key]:
                self._eval_matchup_cache[as_policy_key][against_policy_key] = (payoff, games_played)
            else:
                print(colored(f"Matchup result for \"{as_policy_key}\" vs \"{against_policy_key}\" submitted but it was already cached.", "blue"))

    def _checkpoint_payoff_table(self):
        table_save_file_key = os.path.join(self.payoff_table_save_key_prefix_dir,
                                           f"payoff_table_{self._payoff_table.size()}_polices_"
                                           f"{self._payoff_table.get_num_pending_policies()}_pending_"
                                           f"{gethostname()}_pid_{os.getpid()}_{datetime_str()}.dill")
        table_save_file_path = get_default_path_on_disk_for_minio_key(object_name=table_save_file_key)
        ensure_dir(table_save_file_path)

        self._payoff_table.save_to_dill_file_(save_file_path=table_save_file_path)

        upload_file(storage_client=self._storage_client,
                    bucket_name=self._bucket_name,
                    object_key=table_save_file_key,
                    local_source_path=table_save_file_path)
        self._latest_payoff_table_key = table_save_file_key

        upload_file(storage_client=self._storage_client,
                    bucket_name=self._bucket_name,
                    object_key=self._latest_checkpoint_key,
                    local_source_path=table_save_file_path)
        if CLOUD_PREFIX:
            post_key(storage_client=self._storage_client,
                     bucket_name=self._bucket_name,
                     key=self._latest_checkpoint_key,
                     bulletin_prefix=f"{CLOUD_PREFIX}bulletin")

class ManagerServer(object):

    def __init__(self, config):
        # Stop event for server handler threads to signal this thread that it's time to shutdown
        self._stop_event = threading.Event()
        self._grpc_port = config['grpc_port']
        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=config['num_thread_workers']))

        self._storage_client = connect_storage_client()

        self._root_save_dir = config['logs_and_payoff_table_save_key_prefix'] \
            .replace("DATETIMESTR", datetime_str()) \
            .replace("HOSTNAME", gethostname()) \
            .replace("PID", str(os.getpid()))
        self._root_save_dir = f"{CLOUD_PREFIX}{self._root_save_dir}"

        logger.info(f"root save key prefix is {self._root_save_dir}")

        self._payoff_table_save_dir = os.path.join(self._root_save_dir, "payoff_tables")

        servicer = _PopulationServerServicerImpl(
            stop_event=self._stop_event,
            payoff_table_save_key_prefix_dir=self._payoff_table_save_dir,
            storage_client=self._storage_client,
            bucket_name=BUCKET_NAME,
            max_ping_interval_seconds_to_track_workers=config['max_ping_interval_seconds_to_track_workers'],
            num_games_to_play_for_matchup_evals=config['games_per_eval_matchup'],
            restore_from_payoff_table_key=config['restore_from_payoff_table_key']
        )
        add_PopulationServerServicer_to_server(servicer=servicer, server=self._grpc_server)
        self._grpc_server.add_insecure_port(f'[::]:{self._grpc_port}')

    def run(self):
        logger.info(f'Starting GRPC server. Listening on port {self._grpc_port}.')
        self._grpc_server.start()  # does not block
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt: stopping server")
            self.stop()

    def stop(self):
        self._grpc_server.stop(grace=2)
