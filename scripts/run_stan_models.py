import os, pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import cmdstanpy as cmd
import pandas as pd
import scripts.helper_functions as func

# read data
# what data set are we analysing
dataset = 'exp1'
clean_path = '~/code/ast_human/data/clean/' + dataset + '/'
data = pd.read_csv(clean_path + 'clean_data.csv')

data_dict = func.make_stan_datalist(data, save_trials=0)

BASE_DIR = os.path.expanduser('~/code/ast_human/')

CORE_BUDGET = 22            # 24 physical cores minus 2 for the system
CHAINS = 4
MAX_CONCURRENT = 3          # models running at once; extras queue up

STAN_DIR = os.path.join(BASE_DIR, "scripts/models/stan/safrl")
CSV_ROOT = "/mnt/bigdrive/AST_fits/stan_csv"
FIT_DIR  = os.path.join(BASE_DIR, "model_fits/" + dataset + "/safrl/stan")
COMP_DIR = os.path.join(BASE_DIR, "model_fits/compiled")

# (name, stan file, data, seed)
jobs = [
    ("safrlC",                 "safrlC.stan",                 data_dict, 999),
    ("safrlD cT",              "safrlD_cT.stan",              data_dict, 999),
    ("safrlD decay cT",        "safrlD_decay_cT.stan",        data_dict, 999),
]

n_running = min(MAX_CONCURRENT, len(jobs))
threads_per_chain = max(1, CORE_BUDGET // (n_running * CHAINS))
print(f"{n_running} models at once, {CHAINS} chains x {threads_per_chain} threads each")

print(f"COMP_DIR: {COMP_DIR}")
for name, f, _, _ in jobs:
    exe_path = os.path.join(COMP_DIR, f.replace('.stan', ''))
    print(f"{name}: {exe_path} exists={os.path.exists(exe_path)}")

import shutil

models = {}
for name, f, _, _ in jobs:
    stan_path = os.path.join(STAN_DIR, f)
    exe_path = os.path.join(COMP_DIR, f.replace('.stan', ''))
    
    if os.path.exists(exe_path):
        model = cmd.CmdStanModel(stan_file=stan_path, exe_file=exe_path)
    else:
        # compile (saves next to .stan file)
        model = cmd.CmdStanModel(stan_file=stan_path,
                                 cpp_options={"STAN_THREADS": True})
        src = stan_path.replace('.stan', '')
        shutil.move(src, exe_path)
        # reload from the new location
        model = cmd.CmdStanModel(stan_file=stan_path, exe_file=exe_path)
    
    models[name] = model

def run(name, data, seed):
    csv_dir = os.path.join(CSV_ROOT, name)
    os.makedirs(csv_dir, exist_ok=True)
    fit = models[name].sample(data=data, chains=CHAINS, parallel_chains=CHAINS,
                              threads_per_chain=threads_per_chain,
                              iter_warmup=1000, iter_sampling=1000,
                              adapt_delta=0.95, seed=seed,
                              output_dir=csv_dir, show_progress=True)
    # save as soon as this model finishes, so a later failure doesn't lose it
    os.makedirs(FIT_DIR, exist_ok=True)
    with open(os.path.join(FIT_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(fit, f)
    return fit

fits = {}
with ThreadPoolExecutor(max_workers=n_running) as pool:
    futures = {pool.submit(run, name, data, seed): name for name, _, data, seed in jobs}
    for fut in as_completed(futures):
        name = futures[fut]
        try:
            fits[name] = fut.result()
            print(f"done: {name}")
        except Exception as e:
            print(f"FAILED: {name}\n{e}")