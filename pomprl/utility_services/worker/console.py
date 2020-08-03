import json
import logging
import time

import grpc
from google.protobuf.empty_pb2 import Empty
from minio import Minio

from population_server.cloud_storage import MINIO_DEFAULT_SAVE_PATH
from population_server.protobuf.population_server_pb2 import ManagerStats
from population_server.worker.base_interface import BaseClientManagerInterface, WorkerType, \
    _INFINITE_RETRY_INTERVAL_SECONDS

logger = logging.getLogger(__name__)


class ConsoleManagerInterface(BaseClientManagerInterface):

    def __init__(self,
                 server_host: str,
                 port: int,
                 worker_id: str,
                 minio_client: Minio,
                 minio_bucket_name: str,
                 minio_local_dir: str = MINIO_DEFAULT_SAVE_PATH
                 ):
        super(ConsoleManagerInterface, self).__init__(
            server_host=server_host,
            port=port,
            worker_type=WorkerType.CONSOLE,
            worker_id=worker_id,
            minio_client=minio_client,
            minio_bucket_name=minio_bucket_name,
            minio_local_dir=minio_local_dir)

    def get_manager_stats(self, infinite_retry_on_error: bool = True):
        while True:
            try:
                request = Empty()
                response: ManagerStats = self._stub.GetManagerStats(request)
                break
            except grpc.RpcError as err:
                if infinite_retry_on_error:
                    logger.warning(f"grpc.RPCError raised while getting manager stats:\n{err}\n"
                                   f"(retrying in {_INFINITE_RETRY_INTERVAL_SECONDS} seconds)")
                    time.sleep(_INFINITE_RETRY_INTERVAL_SECONDS)
                else:
                    raise

        stats_dict = json.loads(response.manager_stats_json)
        return stats_dict
