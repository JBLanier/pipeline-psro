import glob
import logging
import os
import re
from typing import Union
import numpy as np
from minio import Minio
from minio.error import NoSuchKey
from mprl.utils import ensure_dir
import mprl
from shutil import copyfile, move, copy2
from filelock import FileLock
# Default save path is the "data" folder in this package
DEFAULT_LOCAL_SAVE_PATH = os.getenv("MPRL_DEFAULT_SAVE_PATH", os.path.join(os.path.dirname(mprl.__file__), "data"))

MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', "LOCAL_STORAGE")
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', None)
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', None)
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", None)

logger = logging.getLogger(__name__)


def connect_storage_client(endpoint: str = MINIO_ENDPOINT, access_key: str = MINIO_ACCESS_KEY, secret_key: str = MINIO_SECRET_KEY):
    if endpoint == "LOCAL_STORAGE":
        logger.info(f"Using local storage instead of Minio (default local data path is {DEFAULT_LOCAL_SAVE_PATH}).")
        return None
    logger.info(f"Connecting Minio bucket storage client to {endpoint} (default local data path is {DEFAULT_LOCAL_SAVE_PATH}).")
    if access_key is None:
        raise ValueError("Must provide a value for access_key if connecting to cloud storage. Passed value is None")
    if secret_key is None:
        raise ValueError("Must provide a value for secret_key if connecting to cloud storage. Passed value is None")
    storage_client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=True)
    return storage_client


def get_default_path_on_disk_for_minio_key(object_name):
    return os.path.join(DEFAULT_LOCAL_SAVE_PATH, object_name)


def key_exists(storage_client: Union[Minio, None], object_name, bucket_name=BUCKET_NAME):
    if storage_client is None:
        return os.path.isfile(get_default_path_on_disk_for_minio_key(object_name=object_name))
    if bucket_name is None:
        raise ValueError("Must specify a bucket name if using cloud storage")
    try:
        return storage_client.stat_object(bucket_name=bucket_name, object_name=object_name) is not None
    except NoSuchKey:
        return False


def copy_object(storage_client: Union[Minio, None], source_object_name, dest_object_name, bucket_name=BUCKET_NAME):
    if storage_client is None:
        src_path = get_default_path_on_disk_for_minio_key(object_name=source_object_name)
        dest_path = get_default_path_on_disk_for_minio_key(object_name=dest_object_name)
        ensure_dir(dest_path)
        return copyfile(src=src_path, dst=dest_path)

    return storage_client.copy_object(bucket_name=bucket_name,
                                      object_name=dest_object_name,
                                      object_source=f"/{bucket_name}/{source_object_name}")


def move_object(storage_client: Union[Minio, None], source_object_name, dest_object_name, bucket_name=BUCKET_NAME):
    if storage_client is None:
        src_path = get_default_path_on_disk_for_minio_key(object_name=source_object_name)
        dest_path = get_default_path_on_disk_for_minio_key(object_name=dest_object_name)
        ensure_dir(dest_path)
        return move(src=src_path, dst=dest_path)

    if bucket_name is None:
        raise ValueError("Must specify a bucket name if using cloud storage")
    assert copy_object(storage_client=storage_client, source_object_name=source_object_name,
                       dest_object_name=dest_object_name, bucket_name=bucket_name)
    storage_client.remove_object(bucket_name=bucket_name, object_name=source_object_name)


def maybe_download_object(storage_client: Union[Minio, None], object_name, bucket_name=BUCKET_NAME, local_directory=DEFAULT_LOCAL_SAVE_PATH,
                          force_download=False):
    save_file_path = os.path.join(local_directory, object_name)
    ensure_dir(save_file_path)
    file_lock = FileLock(lock_file=f"{save_file_path}.lock", timeout=60*5)
    file_lock.acquire()
    try:
        # return if the file is already downloaded
        if os.path.exists(save_file_path) and not force_download:
            logger.debug("{} already exists".format(save_file_path))
            return save_file_path, None

        if storage_client is None:
            default_path = get_default_path_on_disk_for_minio_key(object_name=object_name)
            if not os.path.exists(default_path):
                raise ValueError(f"Using Local Storage and {default_path} doesn't exist")
            if save_file_path == default_path:
                return save_file_path, None
            else:
                copyfile(src=default_path, dst=save_file_path)

        if bucket_name is None:
            raise ValueError("Must specify a bucket name if using cloud storage")

        # ensure the bucket exists
        if not storage_client.bucket_exists(bucket_name=bucket_name):
            raise ValueError("Bucket {} doesn't exist.".join(bucket_name))

        # download the object to the file path
        logger.info("downloading {}".format(object_name))
        object_stat_info = storage_client.fget_object(bucket_name=bucket_name, object_name=object_name, file_path=save_file_path)
    finally:
        file_lock.release()

    return save_file_path, object_stat_info


def maybe_download_objects_with_prefix(storage_client: Union[Minio, None], object_prefix, bucket_name=BUCKET_NAME,
                                       local_directory=DEFAULT_LOCAL_SAVE_PATH, force_download=False):
    file_paths = []
    object_stat_infos = []

    if storage_client is None:
        local_prefix_path = os.path.join(local_directory, object_prefix)
        if not os.path.isdir(local_prefix_path):
            raise ValueError(f"object prefix ({object_prefix}) must denote a directory name if using local storage.")
        for path in glob.glob(pathname=local_prefix_path + "/**", recursive=True):
            if not os.path.isfile(path):
                file_paths.append(path)
                object_stat_infos.append(None)
    else:
        if bucket_name is None:
            raise ValueError("Must specify a bucket name if using cloud storage")
        objects = storage_client.list_objects(bucket_name=bucket_name, prefix=object_prefix, recursive=True)
        for obj in objects:
            if not obj.is_dir:
                file_path, info = maybe_download_object(storage_client=storage_client, object_name=obj.object_name,
                                                        bucket_name=bucket_name, local_directory=local_directory,
                                                        force_download=force_download)
                file_paths.append(file_path)
                object_stat_infos.append(info)

    return file_paths, object_stat_infos


def get_keys_with_prefix(storage_client: Union[Minio, None], object_prefix: str, recurse_into_dirs: bool, bucket_name: str = BUCKET_NAME):
    keys = []
    if storage_client is None:
        local_prefix_path = get_default_path_on_disk_for_minio_key(object_name=object_prefix)
        if not os.path.isdir(local_prefix_path):
            raise ValueError(
                f"object prefix ({object_prefix}) must denote a directory name if using local storage.")
        for path in glob.glob(pathname=local_prefix_path + "/**", recursive=recurse_into_dirs):
            if os.path.isfile(path):
                key = os.path.relpath(path=path, start=local_prefix_path)
                object_name = os.path.join(object_prefix, key)
                keys.append(object_name)
    else:
        if bucket_name is None:
            raise ValueError("Must specify a bucket name if using cloud storage")
        objects = storage_client.list_objects(bucket_name=bucket_name, prefix=object_prefix, recursive=recurse_into_dirs)
        for obj in objects:
            if not obj.is_dir:
                keys.append(obj.object_name)
    return keys


def upload_directory(storage_client: Union[Minio, None], dest_object_key_prefix, local_source_dir, bucket_name=BUCKET_NAME):

    files = glob.glob(local_source_dir + "/**/*", recursive=True)
    results = []

    if storage_client is None:
        local_dest_prefix_path = get_default_path_on_disk_for_minio_key(object_name=dest_object_key_prefix)
        if local_source_dir == local_dest_prefix_path:
            return [None]

    if storage_client is not None and bucket_name is None:
        raise ValueError("Must specify a bucket name if using cloud storage")

    for local_file_path in files:
        if os.path.isfile(local_file_path):
            key = os.path.relpath(path=local_file_path, start=local_source_dir)

            object_name = os.path.join(dest_object_key_prefix, key)

            if storage_client is None:
                dest_file_path = get_default_path_on_disk_for_minio_key(object_name=object_name)
                ensure_dir(dest_file_path)
                copyfile(src=local_file_path, dst=dest_file_path)
                logger.info(f"copied {local_file_path} to {dest_file_path}")
                results.append(dest_file_path)
            else:
                etag = storage_client.fput_object(bucket_name=bucket_name,
                                                  object_name=object_name,
                                                  file_path=local_file_path)
                logger.info(f"uploaded {object_name}")
                results.append(etag)

    return results


def upload_file(storage_client: Union[Minio, None], object_key, local_source_path, bucket_name=BUCKET_NAME):
    if storage_client is not None:
        if bucket_name is None:
            raise ValueError("Must specify a bucket name if using cloud storage")
        etag = storage_client.fput_object(bucket_name=bucket_name,
                                          object_name=object_key,
                                          file_path=local_source_path)
        logger.info("uploaded {}".format(object_key))
    local_dest_path = get_default_path_on_disk_for_minio_key(object_name=object_key)
    if local_source_path == local_dest_path:
        if storage_client is None:
            return local_dest_path
    else:
        ensure_dir(local_dest_path)
        copyfile(src=local_source_path, dst=local_dest_path)
        logger.info(f"copied {local_source_path} to {local_dest_path}")

    if storage_client is None:
        return local_dest_path
    else:
        return etag


def get_tune_sync_to_cloud_fn(storage_client: Union[Minio, None], bucket_name=BUCKET_NAME):
    if storage_client is not None and bucket_name is None:
        raise ValueError("Must specify a bucket name if using cloud storage")

    def sync_to_cloud(local_dir, remote_dir):
        upload_directory(storage_client=storage_client, dest_object_key_prefix=remote_dir, local_source_dir=local_dir,
                         bucket_name=bucket_name)

    return sync_to_cloud


def get_policy_keys_with_prefix_sorted_by_iter(storage_client: Union[Minio, None], object_prefix: str,
                                               recurse_into_dirs: bool, bucket_name: str = BUCKET_NAME,
                                               iter_num_regex="(?<=iter_)\d+", file_ext=".dill"):

    found_keys = get_keys_with_prefix(storage_client=storage_client, object_prefix=object_prefix,
                                      recurse_into_dirs=recurse_into_dirs, bucket_name=bucket_name)

    if len(file_ext) > 0:
        policy_keys = [key for key in found_keys if key.endswith(file_ext)]
    else:
        policy_keys = found_keys

    policy_key_iters = [int(re.search(iter_num_regex, policy_key)[0]) for policy_key in policy_keys]

    policy_keys_sorted_by_iter = [key for key, _ in sorted(zip(policy_keys, policy_key_iters),
                                                           key=lambda pair: pair[1])]

    return policy_keys_sorted_by_iter, sorted(policy_key_iters)


def get_n_evenly_spaced_policy_keys_with_prefix_sorted_by_iter(storage_client: Union[Minio, None], n: int,
                                                               object_prefix: str, recurse_into_dirs: bool,
                                                               bucket_name: str = BUCKET_NAME, iter_num_regex="(?<=iter_)\d+"):
    sorted_policy_keys, iters = get_policy_keys_with_prefix_sorted_by_iter(storage_client=storage_client,
                                                                           object_prefix=object_prefix,
                                                                           recurse_into_dirs=recurse_into_dirs,
                                                                           bucket_name=bucket_name,
                                                                           iter_num_regex=iter_num_regex)
    if n > 1:
        indexes_to_use = np.round(np.linspace(0, len(iters) - 1, n)).astype(int)
    else:
        indexes_to_use = np.asarray(list(range(0, len(iters))))

    filtered_keys = [sorted_policy_keys[idx] for idx in indexes_to_use]
    filtered_iters = [iters[idx] for idx in indexes_to_use]
    return filtered_keys, filtered_iters


def post_key(storage_client: Union[Minio, None], key, bucket_name=BUCKET_NAME, bulletin_prefix="bulletin"):
    new_key = f"{bulletin_prefix}/{key.replace('/','_')}.txt"
    post_local_path = get_default_path_on_disk_for_minio_key(object_name=new_key)
    ensure_dir(post_local_path)
    with open(post_local_path, "+w") as file:
        file.write(key)
    return upload_file(storage_client=storage_client, object_key=new_key, local_source_path=post_local_path,
                       bucket_name=bucket_name)

if __name__ == '__main__':

    storage_client = connect_storage_client()
    keys, iters = get_n_evenly_spaced_policy_keys_with_prefix_sorted_by_iter(storage_client=storage_client, n=6,
                                                                             object_prefix="barrage_human_inits_primary_run/barrage_human_inits_primary_run_10.19.17PM_Jan-24-2020/policy_checkpoints/",
                                                                             recurse_into_dirs=False,
                                                                             bucket_name=BUCKET_NAME)

    for key, iter in zip(keys, iters):
        print(f"{key}, {iter}\n")

