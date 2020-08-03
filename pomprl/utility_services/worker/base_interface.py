import enum
import logging
import threading
import time

import grpc
from google.protobuf.empty_pb2 import Empty
from minio import Minio

from population_server.cloud_storage import MINIO_DEFAULT_SAVE_PATH, connect_minio_client, maybe_download_object
from population_server.payoff_table import PayoffTable, PolicySpec
from population_server.protobuf.population_server_pb2 import PayoffTableKey, WorkerPing, Confirmation, EvalMatchupOrder, EvalResultRequest, EvalMatchupResult
from population_server.protobuf.population_server_pb2_grpc import PopulationServerStub

logger = logging.getLogger(__name__)

_INFINITE_RETRY_INTERVAL_SECONDS = 60
_WORKER_PING_INTERVAL_SECONDS = 20


class FalseConfirmationError(Exception):
    pass


class WorkerType(enum.Enum):
    EVALUATOR = 'evaluator'
    LEARNER = 'learner'
    CONSOLE = 'console'


class BaseClientManagerInterface(object):

    def __init__(self,
                 server_host: str,
                 port: int,
                 worker_type: WorkerType,
                 worker_id: str,
                 minio_client: Minio,
                 minio_bucket_name: str,
                 minio_local_dir: str = MINIO_DEFAULT_SAVE_PATH
                 ):
        self._server_host = server_host
        self._port = port
        self._worker_type = worker_type
        self._worker_id = worker_id
        self._minio_client = minio_client
        self._minio_bucket_name = minio_bucket_name
        self._minio_local_dir = minio_local_dir
        self._stub: PopulationServerStub = self._get_grpc_stub()
        self._worker_ping_thread = self._launch_worker_ping_thread()

    def _get_grpc_stub(self):
        channel = grpc.insecure_channel(target=f"{self._server_host}:{self._port}")
        return PopulationServerStub(channel=channel)

    def _launch_worker_ping_thread(self):
        def worker_ping_loop():
            while True:
                request = WorkerPing()
                request.worker_type = self._worker_type.value
                request.worker_id = self._worker_id
                try:
                    response: Confirmation = self._stub.Ping(request)
                    logger.debug(f"pinged manager, got {response.confirmation}")
                    if not response.confirmation:
                        logger.warning(
                            f"Manager server returned {response.confirmation} in response to our worker ping."
                            f"message: {response.message}")
                except grpc.RpcError as err:
                    logger.warning(f"grpc.RPCError raised while sending worker ping to manager:\n{err}")
                time.sleep(_WORKER_PING_INTERVAL_SECONDS)

        logger.debug(f"starting worker ping thread for worker type '{self._worker_type.value}', id '{self._worker_id}'")
        worker_ping_thread = threading.Thread(target=worker_ping_loop, daemon=True)
        worker_ping_thread.start()
        return worker_ping_thread

    def get_latest_payoff_table(self, infinite_retry_on_error: bool = True):
        while True:
            try:
                request = Empty()
                response: PayoffTableKey = self._stub.GetLatestPayoffTableKey(request)
                break
            except grpc.RpcError as err:
                if infinite_retry_on_error:
                    logger.warning(f"grpc.RPCError raised while getting latest payoff table:\n{err}\n"
                                   f"(retrying in {_INFINITE_RETRY_INTERVAL_SECONDS} seconds)")
                    time.sleep(_INFINITE_RETRY_INTERVAL_SECONDS)
                else:
                    raise
        if response.payoff_table_is_empty:
            logger.debug("Latest payoff table is empty (None)")
            return None, None

        payoff_table_local_path, _ = maybe_download_object(minio_client=self._minio_client,
                                                           bucket_name=self._minio_bucket_name,
                                                           object_name=response.key,
                                                           local_directory=self._minio_local_dir,
                                                           force_download=False)

        latest_payoff_table = PayoffTable.from_dill_file(dill_file_path=payoff_table_local_path)

        return latest_payoff_table, response.key

    def get_size_of_current_payoff_table(self, infinite_retry_on_error: bool = True):
        payoff_table, _ = self.get_latest_payoff_table(infinite_retry_on_error=infinite_retry_on_error)
        if payoff_table is None:
            return 0
        return payoff_table.size()

    def is_policy_key_in_current_payoff_matrix(self, policy_key, infinite_retry_on_error: bool = True):
        payoff_table, _ = self.get_latest_payoff_table(infinite_retry_on_error=infinite_retry_on_error)
        if payoff_table is None:
            # True if there's no payoff table and we aren't looking for anything
            return policy_key is None
        return payoff_table.is_policy_key_in_payoff_matrix(policy_key=policy_key)

    def request_eval_result(self, as_policy_key, as_policy_config_key, as_policy_class_name,
                            against_policy_key, against_policy_config_key, against_policy_class_name,
                            perform_eval_if_not_cached, infinite_retry_on_error: bool = True):

        matchup_request = EvalResultRequest()
        matchup_request.matchup.no_matchups_needed = False
        matchup_request.matchup.as_policy.policy_key = as_policy_key
        matchup_request.matchup.as_policy.policy_model_config_key = as_policy_config_key
        matchup_request.matchup.as_policy.policy_class_name = as_policy_class_name
        matchup_request.matchup.against_policy.policy_key = against_policy_key
        matchup_request.matchup.against_policy.policy_model_config_key = against_policy_config_key
        matchup_request.matchup.against_policy.policy_class_name = against_policy_class_name
        matchup_request.matchup.num_games_to_play = -1
        matchup_request.perform_eval_if_not_cached = perform_eval_if_not_cached

        while True:
            try:
                request = matchup_request
                response: EvalMatchupResult = self._stub.RequestEvalResult(request)
                break
            except grpc.RpcError as err:
                if infinite_retry_on_error:
                    logger.warning(f"grpc.RPCError raised while getting eval result:\n{err}\n"
                                   f"(retrying in {_INFINITE_RETRY_INTERVAL_SECONDS} seconds)")
                    time.sleep(_INFINITE_RETRY_INTERVAL_SECONDS)
                else:
                    raise

        payoff_response = response.payoff
        games_played_response = response.games_played

        if payoff_response == -1:
            payoff_response = None
            games_played_response = None

        return payoff_response, games_played_response


if __name__ == '__main__':
    import os
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

    logging.basicConfig(level=logging.DEBUG)

    minio_client = connect_minio_client(endpoint=MINIO_ENDPOINT,
                                        access_key=MINIO_ACCESS_KEY,
                                        secret_key=MINIO_SECRET_KEY)

    #  test script
    m = BaseClientManagerInterface(server_host='localhost',
                                   port=2727,
                                   worker_type=WorkerType.CONSOLE,
                                   worker_id="lebron",
                                   minio_client=minio_client,
                                   minio_bucket_name=BUCKET_NAME)

    time.sleep(100)

    pass
