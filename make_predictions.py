import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import random as rand
import pickle as pkl
from functions import *
from tensorflow import keras
from itertools import product

AAs = [aa for aa in "GPAVLIMCFYWHKRQNEDST"]
DEFAULT_CHOSEN_MUTS = ["LWKQQR", "DSGERT"]
DEFAULT_POSITIONS = ['D1135', 'S1136', 'G1218', 'E1219', 'R1335', 'T1337']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PAM activity predictions for specified amino acid variants."
    )
    parser.add_argument("--pred-run-name", default="240322_example",
                        help="Name for this prediction run (used to create output subdirectory).")
    parser.add_argument("--saved-model-dir", default="./220924_NN_rand_seed0_ROS",
                        help="Directory containing the trained model to load.")
    parser.add_argument("--output-dir", default="./predictions",
                        help="Root directory where prediction folders will be created.")
    parser.add_argument("--positions-to-randomize", nargs='+', default=DEFAULT_POSITIONS,
                        help="Positions eligible for randomization when --random-muts is supplied.")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of random variants to generate when --random-muts is supplied.")
    parser.add_argument("--no-plots", dest="make_plots", action="store_false",
                        help="Disable plotting of PAM heatmaps.")
    parser.set_defaults(make_plots=True)

    seq_group = parser.add_mutually_exclusive_group()
    seq_group.add_argument("--chosen-muts", nargs='+',
                           help="Six-amino-acid strings defining variants to score.")
    seq_group.add_argument("--chosen-muts-file",
                           help="CSV file (single column) listing variants to score.")
    seq_group.add_argument("--random-muts", action="store_true",
                           help="Randomly generate variants instead of providing them explicitly.")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_variants_from_file(path):
    muts_df = pd.read_csv(path, header=None)
    return [m for m in muts_df[0]]


def validate_positions(requested_positions):
    invalid = set(requested_positions) - set(DEFAULT_POSITIONS)
    if invalid:
        raise ValueError(f"Invalid positions for randomization: {', '.join(sorted(invalid))}")


def build_random_variants(all_positions, positions_to_randomize, n_samples):
    muts = []
    for _ in range(n_samples):
        variant = []
        for pos in all_positions:
            if pos in positions_to_randomize:
                variant.append(rand.choice(AAs))
            else:
                variant.append(pos[0])
        muts.append(variant)
    names = ["".join(name) for name in muts]
    muts_df = pd.DataFrame([list(mut) for mut in muts], columns=all_positions, index=names)
    return muts_df


def plot_predictions(rates_csv, saved_model_dir, output_dir, column_to_append_to_name=None, rates_already_log=True):
    with open(saved_model_dir + "/data_parameters.pkl", 'rb') as read_file:
        vars = pkl.load(read_file)
    run_name, model_name, three_nt_PAM, test_fraction, LOW_RATE_CONSTANT_CUTOFF, LOW_RATE_REPLACE, \
        HEATMAP_MAX, HEATMAP_MIN, nts, all_positions, pams = vars

    ensure_dir(output_dir)

    rates_df = pd.read_csv(rates_csv, index_col=0)
    rates_df = rates_df.reset_index()
    if not rates_already_log:
        rates_df[pams] = np.log10(rates_df[pams])
    for i in range(len(rates_df)):
        row = rates_df.iloc[i,]
        rates = row[pams]
        name = list(row[all_positions])
        name = "".join(name)
        if column_to_append_to_name is not None:
            name_prefix = row[column_to_append_to_name]
            name = name_prefix + "_" + name
        plt.figure(figsize=(5, 1.5))
        ax = plot_heatmap(rates, name, HEATMAP_MAX, HEATMAP_MIN)
        plt.savefig(output_dir + "/" + name + ".svg", format='svg', bbox_inches='tight')
        plt.close()


def main():
    args = parse_args()
    saved_model_dir = args.saved_model_dir
    run_output_root = os.path.join(args.output_dir, args.pred_run_name)
    ensure_dir(args.output_dir)
    ensure_dir(run_output_root)

    chosen_muts = None
    if args.chosen_muts_file:
        chosen_muts = load_variants_from_file(args.chosen_muts_file)
        print("Making predictions for:")
        print(chosen_muts)
    elif args.chosen_muts:
        chosen_muts = args.chosen_muts
    elif not args.random_muts:
        chosen_muts = DEFAULT_CHOSEN_MUTS

    if args.random_muts:
        validate_positions(args.positions_to_randomize)

    with open(saved_model_dir + "/data_parameters.pkl", 'rb') as read_file:
        vars = pkl.load(read_file)
    run_name, model_name, three_nt_PAM, test_fraction, LOW_RATE_CONSTANT_CUTOFF, LOW_RATE_REPLACE, \
        HEATMAP_MAX, HEATMAP_MIN, nts, all_positions, pams = vars
    with open(saved_model_dir + "/mean_std.pkl", 'rb') as read_file:
        DATA_MEAN, DATA_STD = pkl.load(read_file)

    if chosen_muts:
        muts_df = pd.DataFrame([list(mut) for mut in chosen_muts], columns=all_positions, index=chosen_muts)
    else:
        muts_df = build_random_variants(all_positions, args.positions_to_randomize, args.n_samples)

    muts_df = predict_from_saved_model(saved_model_dir, muts_df, DATA_MEAN, DATA_STD, pams)
    preds_csv = os.path.join(run_output_root, "predictions.csv")
    muts_df.to_csv(preds_csv)

    if args.make_plots:
        plot_predictions(preds_csv, saved_model_dir, run_output_root)


if __name__ == "__main__":
    main()

