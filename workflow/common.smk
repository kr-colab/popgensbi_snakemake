"""
Global variables used for naming things downstream
"""

import os

# Project path
N_TRAIN = int(config["n_train"])
RANDOM_SEED = int(config["random_seed"])
TRAIN_SEPARATELY = bool(config["train_embedding_net_separately"])
SIMULATOR = config["simulator"]["class_name"]
PROCESSOR = config["processor"]["class_name"]
EMBEDDING = config["embedding_network"]["class_name"]
UID = f"{SIMULATOR}-{PROCESSOR}-{EMBEDDING}-{RANDOM_SEED}-{N_TRAIN}"
UID += "-sep" if TRAIN_SEPARATELY else "-e2e"
PROJECT_DIR = os.path.join(config["project_dir"], UID)

# Conditional naming of pickled networks
EMBEDDING_NET_NAME = "embedding_network"
NORMALIZING_FLOW_NAME = "normalizing_flow"
if TRAIN_SEPARATELY:
    EMBEDDING_NET_NAME = "pretrain_" + EMBEDDING_NET_NAME
    NORMALIZING_FLOW_NAME = "pretrain_" + NORMALIZING_FLOW_NAME
EMBEDDING_NET = os.path.join(PROJECT_DIR, EMBEDDING_NET_NAME)
NORMALIZING_FLOW = os.path.join(PROJECT_DIR, NORMALIZING_FLOW_NAME)
