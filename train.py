# ---------------------------------------------------------------
# STABILITY TEST FOR IC & SIR INFLUENCE OVER MULTIPLE SEEDS
# WITH FULL LOGGING, CSV EXPORT, AND SUMMARY STORAGE
# ---------------------------------------------------------------
#
# Output files produced:
#   influence_IC_seedX_100runs.csv
#   influence_SIR_seedX_100runs.csv
#   log_seedX.txt
#   full_log_all_seeds.txt
#   results_summary.csv
#
# Perfect for later visualizations.
# ---------------------------------------------------------------

import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import time
import os

# ------------------- SETTINGS -------------------
EDGE_LIST_PATH = "facebook_combined.txt"

RUNS = 100                      # number of simulations per node
SEEDS = [42, 45, 99]           # seeds to test stability
P_IC = 0.1
P_SIR = 0.1

SUMMARY_LIST = []              # stores results for summary CSV
FULL_LOG = []                  # stores logs for all seeds

# Create output folder
OUTDIR = "diffusion_results"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------- LOAD GRAPH -------------------
print("Loading graph...")
G = nx.read_edgelist(EDGE_LIST_PATH, nodetype=int)
nodes = list(G.nodes())
N = len(nodes)

print(f"Loaded graph: {N} nodes, {G.number_of_edges()} edges.\n")


# ---------------- DIFFUSION MODELS -------------------

def ic_diffusion(G, seed, p=0.1):
    active = {seed}
    frontier = {seed}

    while frontier:
        new_frontier = set()
        for node in frontier:
            for neigh in G.neighbors(node):
                if neigh not in active:
                    if random.random() < p:
                        active.add(neigh)
                        new_frontier.add(neigh)
        frontier = new_frontier
    return len(active)


def sir_diffusion(G, seed, p_infect=0.1):
    S, I, R = 0, 1, 2
    state = {n: S for n in G.nodes()}
    state[seed] = I

    infected_now = {seed}
    infected_ever = {seed}

    while infected_now:
        new_infected = set()
        for node in infected_now:
            for neigh in G.neighbors(node):
                if state[neigh] == S and random.random() < p_infect:
                    state[neigh] = I
                    new_infected.add(neigh)
                    infected_ever.add(neigh)
        # recover
        for n in infected_now:
            state[n] = R
        infected_now = new_infected

    return len(infected_ever)


def compute_avg_influence(G, nodes, model_fn, runs, **kwargs):
    influence = {}
    for node in tqdm(nodes):
        spreads = [model_fn(G, node, **kwargs) for _ in range(runs)]
        influence[node] = float(np.mean(spreads))
    return influence


def print_and_log(log_list, text="", end="\n"):
    """Prints and also appends to log list."""
    print(text, end=end)
    log_list.append(text + end)


def log_top_bottom(model_name, seed, influence_dict, log_list):
    sorted_nodes = sorted(influence_dict.items(), key=lambda x: x[1], reverse=True)
    top10 = sorted_nodes[:10]
    bottom10 = sorted_nodes[-10:]

    print_and_log(log_list, "\n==============================")
    print_and_log(log_list, f" {model_name} — Seed {seed} — TOP 10")
    print_and_log(log_list, "==============================")
    for node, spread in top10:
        print_and_log(log_list, f"Node {node}: avg spread = {spread:.2f}")

    print_and_log(log_list, "\n==============================")
    print_and_log(log_list, f" {model_name} — Seed {seed} — BOTTOM 10")
    print_and_log(log_list, "==============================")
    for node, spread in bottom10:
        print_and_log(log_list, f"Node {node}: avg spread = {spread:.2f}")

    # Add to summary list (for CSV)
    for node, spread in top10:
        SUMMARY_LIST.append([seed, model_name, "TOP", node, spread])
    for node, spread in bottom10:
        SUMMARY_LIST.append([seed, model_name, "BOTTOM", node, spread])


# ------------------- MAIN EXPERIMENT -------------------

for seed in SEEDS:
    seed_log = []  # log for this specific seed

    print_and_log(seed_log, "\n" + "=" * 60)
    print_and_log(seed_log, f" RUNNING EXPERIMENTS FOR SEED = {seed}")
    print_and_log(seed_log, "=" * 60)

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # --------------------------------------------------
    # IC MODEL
    # --------------------------------------------------
    print_and_log(seed_log, f"\n[Seed {seed}] Starting IC ({RUNS} runs per node)...")
    t0 = time.time()
    ic_influence = compute_avg_influence(G, nodes, ic_diffusion, RUNS, p=P_IC)
    ic_time = time.time() - t0
    print_and_log(seed_log, f"[Seed {seed}] IC completed in {ic_time/60:.2f} minutes.")

    # Save CSV
    ic_df = pd.DataFrame({"node": list(ic_influence.keys()),
                          "avg_spread": list(ic_influence.values())})
    ic_path = f"{OUTDIR}/influence_IC_seed{seed}_{RUNS}runs.csv"
    ic_df.to_csv(ic_path, index=False)
    print_and_log(seed_log, f"[Seed {seed}] Saved IC results → {ic_path}")

    # Log results
    log_top_bottom("IC", seed, ic_influence, seed_log)


    # --------------------------------------------------
    # SIR MODEL
    # --------------------------------------------------
    print_and_log(seed_log, f"\n[Seed {seed}] Starting SIR ({RUNS} runs per node)...")
    random.seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    sir_influence = compute_avg_influence(G, nodes, sir_diffusion, RUNS, p_infect=P_SIR)
    sir_time = time.time() - t0
    print_and_log(seed_log, f"[Seed {seed}] SIR completed in {sir_time/60:.2f} minutes.")

    # Save CSV
    sir_df = pd.DataFrame({"node": list(sir_influence.keys()),
                           "avg_spread": list(sir_influence.values())})
    sir_path = f"{OUTDIR}/influence_SIR_seed{seed}_{RUNS}runs.csv"
    sir_df.to_csv(sir_path, index=False)
    print_and_log(seed_log, f"[Seed {seed}] Saved SIR results → {sir_path}")

    # Log results
    log_top_bottom("SIR", seed, sir_influence, seed_log)

    # Save seed-specific log
    with open(f"{OUTDIR}/log_seed{seed}.txt", "w") as f:
        f.writelines(seed_log)

    FULL_LOG.extend(seed_log)


# ------------------- SAVE SUMMARY -------------------
summary_df = pd.DataFrame(SUMMARY_LIST,
                          columns=["seed", "model", "group", "node", "avg_spread"])
summary_df.to_csv(f"{OUTDIR}/results_summary.csv", index=False)

with open(f"{OUTDIR}/full_log_all_seeds.txt", "w") as f:
    f.writelines(FULL_LOG)

print("\nAll experiments complete.")
print(f"Results saved inside folder: {OUTDIR}")
