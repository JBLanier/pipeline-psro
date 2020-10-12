import logging
import os
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socket import gethostname
import argparse

import yaml
import pandas as pd
import plotly
import plotly.express as px
from grpc import RpcError
from termcolor import colored

from mprl.utility_services.dashboard.metanash import fictitious_play
from mprl.utility_services.cloud_storage import connect_storage_client, DEFAULT_LOCAL_SAVE_PATH, BUCKET_NAME
from mprl.utility_services.payoff_table import PayoffTable
from mprl.utility_services.utils import datetime_str, pretty_print
from mprl.utility_services.worker.console import ConsoleManagerInterface

WORKER_ID = f"console_{gethostname()}_pid_{os.getpid()}_{datetime_str()}"

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to launch config YAML file", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    logging.basicConfig(level=logging.DEBUG)

    storage_client = connect_storage_client()

    manager_interface = ConsoleManagerInterface(server_host=config['manager_server_host'],
                                                port=config['manager_grpc_port'],
                                                worker_id=WORKER_ID,
                                                storage_client=storage_client,
                                                minio_bucket_name=BUCKET_NAME,
                                                minio_local_dir=DEFAULT_LOCAL_SAVE_PATH)


    def graph_ficticious_play_for_payoff_table(payoff_table: PayoffTable, save_path: str):

        generations = []
        policy_indexes = []
        policy_selection_probs = []
        policy_keys = []
        policy_classes = []
        policy_configs = []
        policy_tags = []
        payoff_matrix = payoff_table.get_payoff_matrix()
        for i, l in enumerate(range(2, len(payoff_matrix) + 1)):
            p = payoff_matrix[:l, :l]
            avgs, exps = fictitious_play(iters=2000, payoffs=p)
            scores = avgs[-1]
            for policy_idx, alpha_rank_score in enumerate(scores):
                policy_spec = payoff_table.get_policy_for_index(policy_idx)
                generations.append(i)
                policy_indexes.append(policy_idx)
                policy_keys.append(policy_spec.key)
                policy_classes.append(policy_spec.class_name)
                policy_configs.append(policy_spec.config_key)
                policy_tags.append(policy_spec.tags)
                policy_selection_probs.append(alpha_rank_score)

        alpha_rank_df = pd.DataFrame({
            "generation": generations,
            "policy": policy_indexes,
            "policy_selection_probs": policy_selection_probs,
            "policy_keys": policy_keys,
            "policy_configs": policy_configs,
            "policy_classes": policy_classes,
            # "policy_tags": ['\n'.join(tags) for tags in policy_tags],
        })

        fig = px.line_3d(alpha_rank_df, x="policy", y="generation", z="policy_selection_probs",
                         color="generation", color_discrete_sequence=px.colors.sequential.Plotly3,
                         template='plotly_dark',
                         title=f"Population Selection Probs (Fictitious Play) over Generations (reload page to refresh)",
                         hover_data=["policy_selection_probs", "policy_classes"],
                         )
        fig.update_layout(
            width=900,
            height=900,
            autosize=True)

        fig.update_layout({
            'plot_bgcolor': 'rgba(34, 34, 34, 1)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })

        fig.update_layout(showlegend=False)
        fig.layout.scene.xaxis.autorange = 'reversed'

        # fig.show()
        plotly.offline.plot(fig, filename=save_path, auto_open=False)


    def periodically_download_and_graph_latest_payoff_table():
        current_payoff_table_key = None
        while True:
            try:
                new_payoff_table, new_payoff_table_key = manager_interface.get_latest_payoff_table(
                    infinite_retry_on_error=False)
            except RpcError as err:
                logger.error(f"Payoff table graphing thread: {err}")
                time.sleep(10)
                continue

            current_graph_path = os.path.join(Path().absolute(), "population_metrics.html")

            if new_payoff_table is None:
                logger.debug("No payoff table is available yet")
                try:
                    os.remove(current_graph_path)
                except OSError:
                    pass
            else:
                if new_payoff_table_key != current_payoff_table_key:
                    logger.debug(f"New payoff table key ({new_payoff_table.size()} policies): {new_payoff_table_key}")
                    graph_ficticious_play_for_payoff_table(payoff_table=new_payoff_table,
                                                           save_path=current_graph_path)
                    current_payoff_table_key = new_payoff_table_key
            time.sleep(10)


    class DashboardHandler(SimpleHTTPRequestHandler):

        def send_stats(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()

            try:
                stats = manager_interface.get_manager_stats(infinite_retry_on_error=False)
            except RpcError as err:
                logger.error(err)
                self.wfile.write(f"(Stats Unavailable)\n{err})".encode("utf8"))
                return

            pending_policy_matchups = stats['stats']['Payoff Table Pending Policy Stats']['pending_policy_matchups']
            pending_policy_matchups_strs = [f"{a}<br>vs<br>{b}" for a, b in pending_policy_matchups]
            del stats['stats']['Payoff Table Pending Policy Stats']['pending_policy_matchups']

            send_html = f"<h2>Manager on {stats['manager_hostname']} ({config['game']})</h2>" \
                        f"<p>{pretty_print(stats['stats']).replace('? ', '')}</p>" \
                        f'<div>' \
                        f'  <p>{pretty_print({f"Pending Policy Matchups ({len(pending_policy_matchups_strs)})": pending_policy_matchups_strs})}</p>' \
                        f'</div>'

            # Send the html message
            self.wfile.write(send_html.encode("utf8"))

        # Handler for the GET requests
        def do_GET(self):
            logger.info(f"GET {self.path}")
            if self.path == '/stats':
                self.send_stats()
            else:
                super(DashboardHandler, self).do_GET()


    host = '0.0.0.0'
    port = config['http_serve_port']

    payoff_table_graphing_thread = threading.Thread(target=periodically_download_and_graph_latest_payoff_table,
                                                    daemon=True)
    payoff_table_graphing_thread.start()

    server = HTTPServer((host, port), DashboardHandler)
    logger.info(f"Starting HTTP server listening on {host}:{port}")

    logger.info(colored(f"Visit http://{gethostname()}:{port} to view dashboard. "
                        f"Refresh the page to see the latest metanash graph (appears after at least 1 policy is added to the payoff table).", "green"))
    # Wait forever for incoming http requests
    server.serve_forever()
