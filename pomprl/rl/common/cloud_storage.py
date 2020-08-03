from minio import Minio
from minio.error import ResponseError, NoSuchKey

import os
import glob
import logging
import re
import numpy as np
from shutil import copyfile
from pomprl.util import ensure_dir
from filelock import FileLock

MINIO_DEFAULT_SAVE_PATH = os.getenv("MINIO_DEFAULT_SAVE_PATH", "/tmp/minio_downloads/")

logger = logging.getLogger(__name__)


def connect_minio_client(endpoint: str, access_key: str, secret_key: str):
    logger.info("Connecting Minio bucket storage client to {}".format(endpoint))
    minio_client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=True)
    return minio_client


def get_default_path_on_disk_for_minio_key(object_name):
    return os.path.join(MINIO_DEFAULT_SAVE_PATH, object_name)


def key_exists(minio_client, bucket_name, object_name):
    try:
        return minio_client.stat_object(bucket_name=bucket_name, object_name=object_name) is not None
    except NoSuchKey:
        return False


def copy_object(minio_client: Minio, bucket_name, source_object_name, dest_object_name):
    return minio_client.copy_object(bucket_name=bucket_name,
                                    object_name=dest_object_name,
                                    object_source=f"/{bucket_name}/{source_object_name}")


def maybe_download_object(minio_client: Minio, bucket_name, object_name, local_directory=MINIO_DEFAULT_SAVE_PATH,
                          force_download=False):
    file_path = os.path.join(local_directory, object_name)
    ensure_dir(file_path)
    file_lock = FileLock(lock_file=f"{file_path}.lock", timeout=60*5)
    with file_lock:
        # return if the file is already downloaded
        if os.path.exists(file_path) and not force_download:
            logger.debug("{} already exists".format(file_path))
            return file_path, None

        # ensure the bucket exists
        if not minio_client.bucket_exists(bucket_name=bucket_name):
            raise ValueError("Bucket {} doesn't exist.".join(bucket_name))

        # download the object to the file path
        logger.info("downloading {}".format(object_name))
        object_stat_info = minio_client.fget_object(bucket_name=bucket_name, object_name=object_name, file_path=file_path)

    return file_path, object_stat_info


def maybe_download_objects_with_prefix(minio_client: Minio, bucket_name, object_prefix,
                                       local_directory=MINIO_DEFAULT_SAVE_PATH, force_download=False):

    objects = minio_client.list_objects(bucket_name=bucket_name, prefix=object_prefix, recursive=True)

    file_paths = []
    object_stat_infos = []

    for obj in objects:
        if not obj.is_dir:
            file_path, info = maybe_download_object(minio_client=minio_client, bucket_name=bucket_name, object_name=obj.object_name,
                                  local_directory=local_directory, force_download=force_download)
            file_paths.append(file_path)
            object_stat_infos.append(info)

    return file_paths, object_stat_infos


def get_keys_with_prefix(minio_client: Minio, bucket_name: str, object_prefix: str, recurse_into_dirs: bool):
    objects = minio_client.list_objects(bucket_name=bucket_name, prefix=object_prefix, recursive=recurse_into_dirs)
    keys = []
    for obj in objects:
        if not obj.is_dir:
            keys.append(obj.object_name)
    return keys

def upload_directory(minio_client: Minio, bucket_name, dest_object_key_prefix, local_source_dir):
    files = glob.glob(local_source_dir+"/**/*", recursive=True)
    etags = []

    for local_file_path in files:

        if os.path.isfile(local_file_path):
            key = os.path.relpath(path=local_file_path, start=local_source_dir)

            object_name = os.path.join(dest_object_key_prefix, key)

            etag = minio_client.fput_object(bucket_name=bucket_name,
                                            object_name=object_name,
                                            file_path=local_file_path)

            logger.info("uploaded {}".format(object_name))

            etags.append(etag)

    return etags


def upload_file(minio_client: Minio, bucket_name, object_key, local_source_path):

    etag = minio_client.fput_object(bucket_name=bucket_name,
                                    object_name=object_key,
                                    file_path=local_source_path)
    copy_path = get_default_path_on_disk_for_minio_key(object_key)
    ensure_dir(copy_path)
    copyfile(src=local_source_path, dst=copy_path)
    logger.info("uploaded {}".format(object_key))

    return etag


def get_tune_sync_to_cloud_fn(minio_client: Minio, bucket_name):

    def sync_to_cloud(local_dir, remote_dir):
        upload_directory(minio_client=minio_client, bucket_name=bucket_name,
                         dest_object_key_prefix=remote_dir, local_source_dir=local_dir)

    return sync_to_cloud


def get_policy_keys_with_prefix_sorted_by_iter(minio_client: Minio, bucket_name: str, object_prefix: str,
                                               recurse_into_dirs: bool, iter_num_regex="(?<=iter_)\d+", file_ext=".dill"):

    found_keys = get_keys_with_prefix(minio_client=minio_client, bucket_name=bucket_name, object_prefix=object_prefix,
                                      recurse_into_dirs=recurse_into_dirs)

    if len(file_ext) > 0:
        policy_keys = [key for key in found_keys if key.endswith(file_ext)]
    else:
        policy_keys = found_keys

    policy_key_iters = [int(re.search(iter_num_regex, policy_key)[0]) for policy_key in policy_keys]

    policy_keys_sorted_by_iter = [key for key, _ in sorted(zip(policy_keys, policy_key_iters),
                                                           key=lambda pair: pair[1])]

    return policy_keys_sorted_by_iter, sorted(policy_key_iters)


def get_n_evenly_spaced_policy_keys_with_prefix_sorted_by_iter(minio_client: Minio, n: int, bucket_name: str, object_prefix: str,
                                               recurse_into_dirs: bool, iter_num_regex="(?<=iter_)\d+"):

    sorted_policy_keys, iters = get_policy_keys_with_prefix_sorted_by_iter(minio_client=minio_client,
                                                                           bucket_name=bucket_name,
                                                                           object_prefix=object_prefix,
                                                                           recurse_into_dirs=recurse_into_dirs,
                                                                           iter_num_regex=iter_num_regex)
    if n > 1:
        indexes_to_use = np.round(np.linspace(0, len(iters) - 1, n)).astype(int)
    else:
        indexes_to_use = np.asarray(list(range(0, len(iters))))

    filtered_keys = [sorted_policy_keys[idx] for idx in indexes_to_use]
    filtered_iters = [iters[idx] for idx in indexes_to_use]
    return filtered_keys, filtered_iters

if __name__ == '__main__':

    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

    minio_client = connect_minio_client(endpoint=MINIO_ENDPOINT,
                                        access_key=MINIO_ACCESS_KEY,
                                        secret_key=MINIO_SECRET_KEY)
    keys, iters = get_n_evenly_spaced_policy_keys_with_prefix_sorted_by_iter(
        minio_client=minio_client,
        n=6,
        bucket_name=BUCKET_NAME,
        object_prefix="barrage_human_inits_primary_run/barrage_human_inits_primary_run_10.19.17PM_Jan-24-2020/policy_checkpoints/",
        recurse_into_dirs=False,
    )

    for key, iter in zip(keys, iters):
        print(f"{key}, {iter}\n")