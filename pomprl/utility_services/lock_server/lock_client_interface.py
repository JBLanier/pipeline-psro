import logging

import grpc
import time
import threading

from population_server.lock_server.protobuf.lock_server_pb2 import LockRequest, LockConfirmation, LockWorkerPing, LockReplaceRequest, NameFilter, LockList
from population_server.lock_server.protobuf.lock_server_pb2_grpc import LockServerStub

logger = logging.getLogger(__name__)

_WORKER_PING_INTERVAL_SECONDS = 15


class LockServerInterface(object):

    def __init__(self, server_host: str, port: int, worker_id: str):
        self._server_host = server_host
        self._port = port
        self._worker_id = worker_id
        self._worker_type = "lock_server_client"
        self._stub: LockServerStub = self._get_grpc_stub()
        self._worker_ping_thread = self._launch_worker_ping_thread()
        self._do_ping()

    def _get_grpc_stub(self):
        channel = grpc.insecure_channel(target=f"{self._server_host}:{self._port}")
        return LockServerStub(channel=channel)

    def _do_ping(self):
        request = LockWorkerPing()
        request.worker_type = self._worker_type
        request.worker_id = self._worker_id
        try:
            response: LockConfirmation = self._stub.Ping(request)
            logger.debug(f"pinged lock server, got {response.confirmation}")
            if not response.confirmation:
                logger.warning(
                    f"Lock Server returned {response.confirmation} in response to our worker ping."
                    f"message: {response.message}")
        except grpc.RpcError as err:
            logger.warning(f"grpc.RPCError raised while sending worker ping to lock server:\n{err}")

    def _launch_worker_ping_thread(self):
        def worker_ping_loop():
            while True:
                self._do_ping()
                time.sleep(_WORKER_PING_INTERVAL_SECONDS)

        logger.debug(f"starting worker ping thread for worker type '{self._worker_type}', id '{self._worker_id}'")
        worker_ping_thread = threading.Thread(target=worker_ping_loop, daemon=True)
        worker_ping_thread.start()
        return worker_ping_thread

    def try_to_reserve_item(self, item_name, remain_after_worker_disconnect=False, request_locks_checkpoint_with_name=None):
        request = LockRequest()
        request.lock_name = item_name
        request.worker_id = self._worker_id
        request.remain_after_disconnect = remain_after_worker_disconnect
        request.request_checkpoint_with_name = request_locks_checkpoint_with_name if request_locks_checkpoint_with_name else ""

        response: LockConfirmation = self._stub.TryToCheckoutLock(request)
        item_reserved: bool = response.confirmation
        return item_reserved

    def try_to_reserve_item_from_list(self, possible_item_names_in_order_of_highest_priority_first, remain_after_worker_disconnect=False):
        for item_name in possible_item_names_in_order_of_highest_priority_first:
            if self.try_to_reserve_item(item_name=item_name,
                                        remain_after_worker_disconnect=remain_after_worker_disconnect):
                return item_name
        return None

    def release_item(self, item_name_to_be_released):
        request = LockRequest()
        request.lock_name = item_name_to_be_released
        request.worker_id = self._worker_id

        response: LockConfirmation = self._stub.ReleaseLock(request)
        item_released: bool = response.confirmation
        return item_released

    def replace_item(self, old_item, new_item, new_item_remains_after_disconnect, request_locks_checkpoint_with_name=None):
        request = LockReplaceRequest()
        request.old_lock_name = old_item
        request.new_lock_name = new_item
        request.worker_id = self._worker_id
        request.new_lock_remains_after_disconnect = new_item_remains_after_disconnect
        request.request_checkpoint_with_name = request_locks_checkpoint_with_name if request_locks_checkpoint_with_name else ""

        response: LockConfirmation = self._stub.ReplaceLock(request)
        item_replaced: bool = response.confirmation
        return item_replaced

    def get_all_items(self, filter_by_string=None):
        request = NameFilter()
        if filter_by_string is not None:
            request.filter = filter_by_string
        else:
            request.filter = ""
        response: LockList = self._stub.GetAllLocksWithString(request)
        return list(response.lock_names)

    def _are_all_barrier_members_at_barrier(self, barrier_name):
        barrier_member_locks = self.get_all_items(filter_by_string=f"b {barrier_name} m")
        members_at_barrier = []
        all_members_are_at_barrier = True
        for member_lock in barrier_member_locks:
            lock_components = member_lock.split(' ')
            _, _bar_name, _, member_name, at_barrier = lock_components
            if at_barrier == 'T':
                members_at_barrier.append(member_name)
                continue
            elif at_barrier == 'F':
                all_members_are_at_barrier = False
            else:
                raise ValueError(f"Failed to correctly parse barrier lock {member_lock} when split into components {lock_components}")
        return all_members_are_at_barrier, members_at_barrier

    def join_barrier_group(self, barrier_name, member_name, grace_period_for_others_to_join_s=0.0):
        # Quickly made multi-node barrier using the lock server
        # All barrier logic is in this function, no central manager.
        # Assumes barrier members reenter this barrier at a frequency slower than 1Hz or so,
        # otherwise race conditions can occur
        assert ' ' not in barrier_name
        assert ' ' not in member_name
        member_not_at_barrier_lock = f"b {barrier_name} m {member_name} F"
        member_at_barrier_lock = f"b {barrier_name} m {member_name} T"
        barrier_blocking_lock = f"b {barrier_name} T"
        assert self.try_to_reserve_item(item_name=member_not_at_barrier_lock, remain_after_worker_disconnect=False)

        if grace_period_for_others_to_join_s > 0.0:
            time.sleep(grace_period_for_others_to_join_s)

        def wait_at_barrier():
            self.try_to_reserve_item(item_name=barrier_blocking_lock, remain_after_worker_disconnect=True)
            assert self.replace_item(old_item=member_not_at_barrier_lock, new_item=member_at_barrier_lock,
                                     new_item_remains_after_disconnect=False)
            while True:
                all_members_are_at_barrier, members_at_barrier = self._are_all_barrier_members_at_barrier(barrier_name=barrier_name)
                if all_members_are_at_barrier:
                    self.release_item(item_name_to_be_released=barrier_blocking_lock)
                    break
                elif len(self.get_all_items(filter_by_string=barrier_blocking_lock)) == 0:
                    break
                else:
                    time.sleep(0.2)
                    print(f"waiting at barrier ({members_at_barrier})")
            assert self.replace_item(old_item=member_at_barrier_lock, new_item=member_not_at_barrier_lock,
                                     new_item_remains_after_disconnect=False)

        def leave_barrier_group():
            self.replace_item(old_item=member_not_at_barrier_lock, new_item=member_at_barrier_lock,
                              new_item_remains_after_disconnect=False)
            self.release_item(item_name_to_be_released=member_at_barrier_lock)

        return wait_at_barrier, leave_barrier_group

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    l1 = LockServerInterface(server_host="localhost",
                             port=3737,
                             worker_id="worker1")
    print(l1.try_to_reserve_item("this mixtape"))

    l2 = LockServerInterface(server_host="localhost",
                             port=3737,
                             worker_id="worker2")
    print(l2.try_to_reserve_item("this mixtape"))

    print(l2.try_to_reserve_item("another mixtape"))

    print(l2.try_to_reserve_item_from_list(["this mixtape", "another mixtape", "a third mixtape", "a fourth mixtape"]))

    # time.sleep(30)
    # l1.release_item("this mixtape")
