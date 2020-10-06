#!/usr/bin/env python
import os

import argparse
import subprocess
from termcolor import colored

GCLOUD_PROJECT_NAME = os.environ["GCLOUD_PROJECT_NAME"]

WORKER_NUM_AND_GAME_TO_MACHINE_TYPE = {
    (1, "fives"): "custom-6-5632",
    (1, "kuhn_poker"): "custom-4-5120",
    (3, "fives"): "custom-16-14848",
    (3, "kuhn_poker"): "custom-10-32768",
}
_to_add = {}
for k, v in WORKER_NUM_AND_GAME_TO_MACHINE_TYPE.items():
    if k[1] == "kuhn_poker":
        _to_add[(k[0], "leduc_poker")] = v
WORKER_NUM_AND_GAME_TO_MACHINE_TYPE.update(_to_add)


def launch_gc_instance(po_exp_command):
    #for example, a valid command is ./launch_exp.sh pipe.sh 1 3 3 kuhn_poker

    command_elements = po_exp_command.split(' ')
    command = command_elements[0]
    args = command_elements[1:]
    exp_config, seed_num, num_workers, num_eval_workers, poker_game_version = args
    instance_name = f"{exp_config.replace('.sh','')}-{seed_num}-{num_workers}-{num_eval_workers}-{poker_game_version.replace('_','-')}-v2"
    print(f"instance name will be {instance_name}")
    container_args_str = ""
    for arg in args:
        container_args_str += f"--container-arg={arg} "

    command = f"gcloud beta compute " \
        f"--project={GCLOUD_PROJECT_NAME} instances create-with-container {instance_name} " \
        f"--zone=us-central1-a " \
        f"--machine-type={WORKER_NUM_AND_GAME_TO_MACHINE_TYPE[(int(num_workers), poker_game_version)]} " \
        f"--subnet=default " \
        f"--network-tier=PREMIUM " \
        f"--metadata=google-logging-enabled=true " \
        f"--maintenance-policy=MIGRATE " \
        f"--service-account=862565516536-compute@developer.gserviceaccount.com " \
        f"--scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append " \
        f"--tags=http-server,https-server " \
        f"--image=cos-stable-81-12871-119-0 " \
        f"--image-project=cos-cloud " \
        f"--boot-disk-size=30GB " \
        f"--boot-disk-type=pd-standard " \
        f"--boot-disk-device-name={instance_name} " \
        f"--container-image=gcr.io/{GCLOUD_PROJECT_NAME}/po-docker " \
        f"--container-restart-policy=always " \
        f"--container-tty " \
        f"--container-command={command} " \
        f"{container_args_str}" \
        f"--container-env=COLUMNS=230,LINES=59 " \
        f"--labels=container-vm=cos-stable-81-12871-119-0"

    print(f"\n\nRunning command:\n{command}\n\n")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    assert process.returncode == 0, f"process returncode was {process.returncode}"
    print(f"done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--commands", help="path one docker command per line", type=str, required=True)
    args = parser.parse_args()
    commands_path = args.commands

    with open(commands_path, "r") as file:
        lines = file.readlines()
    commands = []
    for line in lines:
        if line:
            assert "./launch_exp.sh" in line
            commands.append(line.replace("\n", ""))

    commands_str = ""
    for cmd in commands:
        commands_str += f"\n{cmd}"
    response = input(f"Your about to launch {len(commands)} instances with these commands:\n(begin commands){commands_str}\n(end commands)\nIs the correct? (y to continue): ")
    if response != 'y':
        print("exiting")
        exit()
    response2 = input("Hey, are you SURE you want to launch these? This costs real money! (Y to continue): ")
    if response2 != 'Y':
        print("exiting")
        exit()

    print("Launching...")
    for cmd in commands:
        launch_gc_instance(po_exp_command=cmd)

    print(colored("Done with all instance launches.", "green"))