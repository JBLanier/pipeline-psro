from mprl.utility_services.lock_server.protobuf.lock_server_pb2 import LockRequest, LockConfirmation, LockWorkerPing, LockList
from mprl.utility_services.lock_server.protobuf.lock_server_pb2_grpc import LockServerServicer, \
    add_LockServerServicer_to_server

import logging
import threading
import os
from socket import gethostname
import grpc
import time
from concurrent import futures
from queue import PriorityQueue, Empty
import argparse
import yaml
from termcolor import colored
from mprl.utility_services.cloud_storage import connect_storage_client, DEFAULT_LOCAL_SAVE_PATH, get_default_path_on_disk_for_minio_key, upload_file, post_key, BUCKET_NAME
from mprl.utility_services.utils import datetime_str, pretty_print, ensure_dir
from mprl.utility_services.worker.console import ConsoleManagerInterface
import dill

WORKER_ID = f"lock_manager_{gethostname()}_pid_{os.getpid()}_{datetime_str()}"

CLOUD_PREFIX = os.getenv("CLOUD_PREFIX", "")
LATEST_CHECKPOINT_KEY = f"{CLOUD_PREFIX}lock_server_checkpoints/{WORKER_ID}/latest.dill"

WORKER_PING_TOLERANCE_SECONDS = 30

logger = logging.getLogger(__name__)


def _string_with_dict_items_one_per_line(set):
    out = ""
    for key, val in set.items():
        out += f"{key}: {val}\n"
    return out


class LockServerServicerImpl(LockServerServicer):

    def __init__(self, storage_client, bucket_name):

        self._item_locks_modification_lock = threading.RLock()
        self._item_locks = {}

        self._worker_ping_modification_lock = threading.RLock()
        self._recent_worker_pings = PriorityQueue()
        self._start_time = time.time()
        self._max_ping_interval_seconds_to_track_workers = WORKER_PING_TOLERANCE_SECONDS
        self._storage_client = storage_client
        self._bucket_name = bucket_name
        self._time_of_last_checkpoint = time.time()

    def TryToCheckoutLock(self, request: LockRequest, context):
        lock_name = request.lock_name
        worker_id = request.worker_id
        remain_after_disconnect = request.remain_after_disconnect

        with self._worker_ping_modification_lock:
            self._unsafe_remove_old_worker_pings()

        with self._item_locks_modification_lock:
            if lock_name not in self._item_locks:
                self._item_locks[lock_name] = (worker_id, remain_after_disconnect)
                lock_checkout_was_successful = True
                msg = f"lock checked out ({lock_name})"
                logger.info(colored(f"Checked out lock \"{lock_name}\"","green"))
                logger.info(f"Current locks:\n{_string_with_dict_items_one_per_line(self._item_locks)}")
                if request.request_checkpoint_with_name:
                    self._save_checkpoint(checkpoint_name=f"{request.request_checkpoint_with_name}")
                self._save_checkpoint_if_at_time_interval()
            else:
                lock_checkout_was_successful = False
                msg = f"lock is already reserved ({lock_name})"
                logger.info(f"tried to check out reserved lock {lock_name}")

        response = LockConfirmation()
        response.confirmation = lock_checkout_was_successful
        response.message = msg
        return response

    def ReleaseLock(self, request: LockRequest, context):
        lock_name = request.lock_name

        with self._item_locks_modification_lock:
            if lock_name in self._item_locks:
                del self._item_locks[lock_name]
                lock_release_was_successful = True
                msg = f"lock released ({lock_name})"
                logger.info(colored(f"Released lock \"{lock_name}\"", "blue"))
                logger.info(f"Current locks:\n{_string_with_dict_items_one_per_line(self._item_locks)}")
                self._save_checkpoint_if_at_time_interval()
            else:
                lock_release_was_successful = False
                msg = f"lock is not reserved to nothing was done ({lock_name})"

        response = LockConfirmation()
        response.confirmation = lock_release_was_successful
        response.message = msg
        return response

    def ReplaceLock(self, request, context):
        old_lock_name = request.old_lock_name
        new_lock_name = request.new_lock_name
        worker_id = request.worker_id
        new_lock_remains_after_disconnect = request.new_lock_remains_after_disconnect

        with self._worker_ping_modification_lock:
            self._unsafe_remove_old_worker_pings()

        with self._item_locks_modification_lock:
            if old_lock_name in self._item_locks:
                if new_lock_name not in self._item_locks:
                    del self._item_locks[old_lock_name]
                    self._item_locks[new_lock_name] = (worker_id, new_lock_remains_after_disconnect)
                    lock_replace_was_successful = True
                    msg = "lock replaced"
                    logger.info(colored(f"Replace lock: old was \"{old_lock_name}\"\nnew is \"{new_lock_name}\"", "yellow"))
                    logger.info(f"Current locks:\n{_string_with_dict_items_one_per_line(self._item_locks)}")
                    if request.request_checkpoint_with_name:
                        self._save_checkpoint(checkpoint_name=f"{request.request_checkpoint_with_name}")
                    self._save_checkpoint_if_at_time_interval()
                else:
                    lock_replace_was_successful = False
                    msg = f"lock is already reserved ({new_lock_name})"
                    logger.info(colored(f"tried to check out (in replacement) reserved lock {new_lock_name}", "yellow"))
            else:
                lock_replace_was_successful = False
                msg = f"lock isn't reserved ({old_lock_name})"
                logger.info(colored(f"tried to replace nonexistant lock {old_lock_name}", "yellow"))

        response = LockConfirmation()
        response.confirmation = lock_replace_was_successful
        response.message = msg
        return response

    def GetAllLocksWithString(self, request, context):
        filter = request.filter

        with self._worker_ping_modification_lock:
            self._unsafe_remove_old_worker_pings()

        matching_locks = []
        with self._item_locks_modification_lock:
            for lock_name in self._item_locks.keys():
                if filter in lock_name:
                    matching_locks.append(lock_name)

        response = LockList()
        response.lock_names.extend(matching_locks)
        return response

    def _unsafe_remove_old_worker_pings(self):
        logger.debug("removing old worker pings")
        if not self._recent_worker_pings.empty():

            worker_ids_before_removal = set()
            for ping_time, worker_ping in list(self._recent_worker_pings.queue):
                worker_ids_before_removal.add(worker_ping.worker_id)

            while True:
                try:
                    ping_time, worker_ping = self._recent_worker_pings.get_nowait()
                except Empty:
                    break
                logger.debug(f"oldest ping is {time.time() - ping_time} seconds old")
                if time.time() - ping_time < self._max_ping_interval_seconds_to_track_workers:
                    # put it back if it's not too old
                    self._recent_worker_pings.put((ping_time, worker_ping))
                    break

            worker_ids_after_removal = set()
            for ping_time, worker_ping in list(self._recent_worker_pings.queue):
                worker_ids_after_removal.add(worker_ping.worker_id)

            logger.debug(f"Connected workers ({len(worker_ids_after_removal)}):\n{worker_ids_after_removal}")

            with self._item_locks_modification_lock:
                locks_changed = self._unsafe_clear_orphaned_locks(all_connected_worker_ids=worker_ids_after_removal)
                if locks_changed:
                    logger.info(f"Current locks:\n{_string_with_dict_items_one_per_line(self._item_locks)}")
        logger.debug("done removing old worker pings")


    # def _unsafe_clear_locks_belonging_to_worker(self, worker_id):
    #     # dont use this without securing self._item_locks_modification_lock first
    #     removal_count = 0
    #     for k, v in list(self._item_locks.items()):
    #         if worker_id == v:
    #             del self._item_locks[k]
    #             removal_count += 1
    #     logger.info(f"Removed all {removal_count} locks belonging to {worker_id}")

    def _unsafe_clear_orphaned_locks(self, all_connected_worker_ids):
        # dont use this without securing self._item_locks_modification_lock first
        anything_removed = False
        for k, (worker, remain_after_disconnect) in list(self._item_locks.items()):
            if worker not in all_connected_worker_ids and not remain_after_disconnect:
                del self._item_locks[k]
                anything_removed = True
                logger.info(colored(f"Removed orphaned lock \"{k}\" that belonged to {worker}", "white"))
        return anything_removed

    def Ping(self, request: LockWorkerPing, context):
        logger.debug(f"Got worker ping for {request.worker_type}, worker id {request.worker_id}")
        with self._worker_ping_modification_lock:
            self._recent_worker_pings.put((time.time(), request))
            self._unsafe_remove_old_worker_pings()

        response = LockConfirmation()
        response.confirmation = True
        return response

    def _save_checkpoint_if_at_time_interval(self):
        now = time.time()
        with self._item_locks_modification_lock:
            if now - self._time_of_last_checkpoint > 60*60*4:  # 4 hours
                checkpoint_name = f"locks_checkpoint_{len(self._item_locks)}_locks_{datetime_str()}"
                self._save_checkpoint(checkpoint_name=checkpoint_name)
                self._time_of_last_checkpoint = now

    def _save_checkpoint(self, checkpoint_name):
        checkpoint_key = f"{CLOUD_PREFIX}lock_server_checkpoints/{WORKER_ID}/{checkpoint_name}_utc_{time.time()}.dill"
        checkpoint_path = get_default_path_on_disk_for_minio_key(checkpoint_key)
        ensure_dir(checkpoint_path)
        with open(checkpoint_path, "wb+") as dill_file:
            dill.dump(self._item_locks, dill_file)
        upload_file(storage_client=self._storage_client,
                    bucket_name=self._bucket_name,
                    object_key=checkpoint_key,
                    local_source_path=checkpoint_path)
        logger.info(colored(f"Saved locks checkpoint to {checkpoint_key}", "magenta"))

        upload_file(storage_client=self._storage_client,
                    bucket_name=self._bucket_name,
                    object_key=LATEST_CHECKPOINT_KEY,
                    local_source_path=checkpoint_path)
        logger.info(colored(f"Also Saved locks checkpoint to {LATEST_CHECKPOINT_KEY}", "magenta"))
        if CLOUD_PREFIX:
            post_key(storage_client=self._storage_client,
                     bucket_name=self._bucket_name,
                     key=LATEST_CHECKPOINT_KEY,
                     bulletin_prefix=f"{CLOUD_PREFIX}bulletin")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to launch config YAML file", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    logging.basicConfig(level=logging.INFO)

    logger.info(colored(f"Latest lock server checkpoint will always be at {LATEST_CHECKPOINT_KEY}", 'yellow'))

    logger.info(f"launching with config:\n{pretty_print(config)}")

    storage_client = connect_storage_client()

    manager_interface = ConsoleManagerInterface(server_host=config['manager_server_host'],
                                                port=config['manager_grpc_port'],
                                                worker_id=WORKER_ID,
                                                storage_client=storage_client,
                                                minio_bucket_name=BUCKET_NAME,
                                                minio_local_dir=DEFAULT_LOCAL_SAVE_PATH)

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    lock_servicer = LockServerServicerImpl(storage_client=storage_client, bucket_name=BUCKET_NAME)
    add_LockServerServicer_to_server(servicer=lock_servicer, server=grpc_server)
    grpc_server.add_insecure_port(f"[::]:{config['locker_server_grpc_port']}")

    logger.info(f"Starting GRPC server. Listening on port {config['locker_server_grpc_port']}.")
    grpc_server.start()  # does not block
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: stopping server")
        grpc_server.stop(grace=2)
