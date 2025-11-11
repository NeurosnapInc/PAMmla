import argparse
import numpy as np
import pandas as pd
import os
import pickle as pkl
from functions import *
from tensorflow import keras
from itertools import product
import random
import warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

warnings.filterwarnings("ignore")

DEFAULT_SAVED_MODEL_DIRS = ["./220924_NN_rand_seed0_ROS",
                            "./220924_NN_rand_seed2_ROS",
                            "./220924_NN_rand_seed3_ROS"]
DEFAULT_STARTING_VARIANTS = ["DSGERT"]
DEFAULT_PAMS_TO_MAX_SELECTIVITY = ["NGTG", "NGTA", "NGTT", "NGTC"]
AAs = [aa for aa in "GPAVLIMCFYWHKRQNEDST"]
DATA_MEANS = []
DATA_STDS = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run in silico directed evolution with the PAMmla model."
    )
    parser.add_argument("--pred-run-name", default="NGT_selectivity_example",
                        help="Name for this evolution run (used in the output path).")
    parser.add_argument("--saved-model-dir", dest="saved_model_dirs", action="append",
                        help="Directory containing a trained model. Provide multiple times to average predictions.")
    parser.add_argument("--output-dir", default=".",
                        help="Base directory where results will be written.")
    parser.add_argument("--save-traj", action="store_true",
                        help="Write CSV snapshots for every selection round.")

    parser.add_argument("--mutations-per-variant", type=int, default=4,
                        help="Number of mutations introduced per variant each round.")
    parser.add_argument("--variants-per-round", type=int, default=1000,
                        help="Number of variants generated per round.")
    parser.add_argument("--n-best-variants-per-round", type=int, default=10,
                        help="Variants retained between rounds.")
    parser.add_argument("--decay-after-n-rounds-plateau", type=int, default=1,
                        help="Rounds of plateau before reducing mutations_per_variant.")
    parser.add_argument("--ramp-up-rounds", type=int, default=0,
                        help="Rounds used to gradually apply selection pressure.")

    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument("--starting-variants", nargs='+',
                             help="Six-amino-acid strings used to seed evolution.")
    start_group.add_argument("--starting-variants-file",
                             help="CSV file (single column) listing starting variants.")
    start_group.add_argument("--random-starting-variants", action="store_true",
                             help="Start from randomly generated variants.")

    parser.add_argument("--pam-to-max",
                        help="PAM to maximize activity on.")
    parser.add_argument("--pam-to-min",
                        help="PAM to minimize activity on.")
    parser.add_argument("--pams-to-max-selectivity", nargs='+',
                        help="List of PAMs to maximize selectivity for. Pass 'None' to disable.")
    parser.add_argument("--high-pams", nargs='+', default=[],
                        help="PAMs required to exceed the corresponding high cutoffs.")
    parser.add_argument("--high-cutoffs", nargs='+', type=float, default=[],
                        help="Log10 cutoffs for PAMs listed in --high-pams.")
    parser.add_argument("--low-pams", nargs='+', default=[],
                        help="PAMs required to stay below the corresponding low cutoffs.")
    parser.add_argument("--low-cutoffs", nargs='+', type=float, default=[],
                        help="Log10 cutoffs for PAMs listed in --low-pams.")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_variants_from_file(path):
    muts_df = pd.read_csv(path, header=None)
    return [m for m in muts_df[0]]


def validate_cutoff_lists(pams, cutoffs, label):
    if len(pams) != len(cutoffs):
        raise ValueError(f"{label} PAMs and cutoffs must have the same length.")


def normalize_pams_to_max_selectivity(values):
    if values is None:
        return None
    if len(values) == 1 and values[0].lower() == "none":
        return None
    return values


def prepare_starting_variants(args, aa_positions):
    if args.random_starting_variants:
        return 'random'
    if args.starting_variants_file:
        variants = load_variants_from_file(args.starting_variants_file)
    elif args.starting_variants:
        variants = args.starting_variants
    else:
        variants = DEFAULT_STARTING_VARIANTS
    expected_len = len(aa_positions)
    invalid = [seq for seq in variants if len(seq) != expected_len]
    if invalid:
        raise ValueError(f"All starting variants must be {expected_len} amino acids long. "
                         f"Invalid entries: {', '.join(invalid)}")
    return [list(seq) for seq in variants]


def mutate(variants, allowed_aas_list, mutations_per_variant, n_output_variants):
    """Takes a variant (list) and mutates random positions at a time
    Returns list of length n_output_variants of randomly mutated variants.
    """
    output = []
    for variant in variants:
        for _ in range(n_output_variants):
            new_var = variant[:]
            for _ in range(mutations_per_variant):
                pos = random.choice(range(len(variant)))
                new_var[pos] = random.choice(allowed_aas_list[pos])
            output.append(new_var)
    return np.array(output)


def ramp_cutoffs(original_cutoff, current_round, ramp_up_rounds, type='increase'):
    """reduce the cutoff if ramp up rounds is > 1 to reduce selection pressure in first few rounds"""
    assert(type == 'increase' or type == 'decrease')
    if current_round >= ramp_up_rounds:
        return original_cutoff
    elif type == 'increase':
        diff = original_cutoff + 5 # positive number
        step = diff/ramp_up_rounds
        remove = (ramp_up_rounds - current_round) * step
        return original_cutoff - remove
    elif type == 'decrease':
        diff = -1*original_cutoff - 1.5 # positive number
        step = diff / ramp_up_rounds
        add = (ramp_up_rounds - current_round) * step
        return original_cutoff + add


def select_custom(rates_df,
                  n_best,
                  pam_to_max,
                  pam_to_min,
                  high_pams,
                  high_cutoffs,
                  low_pams,
                  low_cutoffs,
                  pams_to_max_selectivity,
                  ramp_up_rounds,
                  curr_round,
                  all_pams):
    """Chooses variants with rates for all pams specified are greater than a cutoff, smaller than a cutoff, maximized, or minimized.
    Only one PAM can be either maximized or minimized. Arbitrary numbers of pams can be set to a threshhold.
    :parameter: pam (tuple of form:)
    (pam to maximize, pam to minimize, [list of high PAMs], [list of k minimums for high PAMs], [list of low PAMs], [list of k maximums for low PAMs])
    """

    assert (pam_to_max is not None) or (pam_to_min is not None) or (pams_to_max_selectivity is not None)# Need to provide a PAM to maximize or minimize"
    assert (len(high_pams) == len(high_cutoffs))
    assert (len(low_pams) == len(low_cutoffs))
    rates_df_copy = rates_df.copy()
    if pams_to_max_selectivity is not None:
        all_rates_sum = np.sum(np.power(10, rates_df_copy[all_pams]), axis=1)
        rates_of_interest_sum = np.sum(np.power(10, rates_df_copy[pams_to_max_selectivity]), axis=1)
        selectivity = np.divide(rates_of_interest_sum, all_rates_sum)
        rates_df_copy["sorted_on"] = selectivity
    elif (pam_to_max is not None) and (pam_to_min is not None): # if max and min pams are both provided then divide max by min
        rates_df_copy["sorted_on"] = np.divide(np.power(10, rates_df_copy[pam_to_max]), np.power(10, rates_df_copy[pam_to_min]))
    elif pam_to_max is not None:  # otherwise just maximize or minimize
        rates_df_copy["sorted_on"] = rates_df_copy[pam_to_max]
    elif pam_to_min is not None:  # PAM to minimize so multiply by negative 1
        rates_df_copy["sorted_on"] = np.multiply(rates_df_copy[pam_to_min], -1)

    # remove variants that do not meet the cutoffs before sorting
    for i in range(len(high_pams)):
        pam = high_pams[i]
        cutoff = ramp_cutoffs(high_cutoffs[i], curr_round, ramp_up_rounds, type='increase')
        print("Cutoff for " + pam + ":", cutoff)
        rates_df_copy = rates_df_copy.drop(rates_df_copy[rates_df_copy[pam] < cutoff].index)
    for i in range(len(low_pams)):
        pam = low_pams[i]
        cutoff = ramp_cutoffs(low_cutoffs[i], curr_round, ramp_up_rounds, type='decrease')
        print("Cutoff for " + pam + ":", cutoff)
        rates_df_copy = rates_df_copy.drop(rates_df_copy[rates_df_copy[pam] > cutoff].index)

    if len(rates_df_copy) == 0:
        print("Oops you killed all the variants! Your selection conditions may be too stringent, or you may have began with a poor starting sequence.")
        exit()

    best = rates_df_copy.nlargest(n_best, columns="sorted_on", keep='first')
    best = best.reset_index(drop=True)
    highest_score = best["sorted_on"][0]
    return (best, highest_score)


def select_highest_rates(rates_df, n_best, pam, pam_against, pams):  # leaving pams & pam against as an argument even though doesnt use to keep in same format as selectivity function
    """Takes a dataframe of rates and returns the n columns with highest rates on pam
    Assumes the columns are named after PAMs"""
    rates_df_copy = rates_df.copy()
    rates_df_copy["sorted_on"] = rates_df_copy[pam]
    best = rates_df_copy.nlargest(n_best, columns="sorted_on", keep='first')
    best = best.reset_index(drop=True)
    highest_rate = best["sorted_on"][0]
    return (best, highest_rate)


def select_most_selective(rates_df,
                          n_best,
                          pam_to_max,
                          pam_to_min,
                          high_pams,
                          high_cutoffs,
                          low_pams,
                          low_cutoffs,
                          ramp_up_rounds,
                          curr_round,
                          all_pams): # doesnt use PAM against
    """Takes a dataframe of rates and returns the n columns with most selective variants for pam
    Assumes the columns are named after PAMs.
    Assumes the rates are in log values so exponentiates before summing"""
    all_rates_sum = np.sum(np.power(10, rates_df[all_pams]), axis=1)
    selectivity = np.divide(np.power(10, rates_df[pam_to_max]), all_rates_sum)
    rates_df_copy = rates_df.copy()
    rates_df_copy["sorted_on"] = selectivity
    best = rates_df_copy.nlargest(n_best, columns="sorted_on", keep="first")
    best = best.reset_index(drop=True)
    highest_selectivity = best["sorted_on"][0]
    return (best, highest_selectivity)

def generate_random_variant(allowed_aas_list):
    var = []
    for i in range(len(allowed_aas_list)):
        var.append(random.choice(allowed_aas_list[i]))
    return var

def evolve(selection_function,
           mutations_per_variant,
           variants_per_round,
           n_best_variants_per_round,
           decay_after_n_rounds_plateau,
           saved_model_dirs,
           all_pams,
           aa_positions,
           pam_to_max=None,
           pam_to_min=None,
           high_pams=[],
           high_cutoffs=[],
           low_pams=[],
           low_cutoffs=[],
           pams_to_max_selectivity=None,
           starting_variants='random',
           ramp_up_rounds=0 # number of rounds at beginning to take to ramp up selection to full strength
           ):
    """Evolve variants from the list of allowed amino acids at each position
    Fitness of each variant is tested with fitness_function.
    PAM profile for each variant is predicted using average of PAM prediction models"""
    plateau_rounds = 0
    curr_high_score = None
    trajectory = []
    curr_round = 0
    curr_best_df = pd.DataFrame()
    while mutations_per_variant > 0:
        curr_round += 1
        print("Evolution round: ", curr_round)
        print("Curr high score: ", curr_high_score)
        print("Mutations per variant: ", mutations_per_variant)

        allowed_aas_list = []
        for pos in aa_positions:
            allowed_aas_list.append(AAs)

        if starting_variants == 'random':
            starting_variants = []
            for _ in range(variants_per_round):
                v = generate_random_variant(allowed_aas_list)
                starting_variants.append(v)

        if starting_variants == 'random' and curr_round == 1:
            mutants = starting_variants
        else:
            mutants = mutate(starting_variants, allowed_aas_list, mutations_per_variant, variants_per_round)
        mutants = pd.DataFrame(mutants, columns=aa_positions)

        # Make predictions from all saved models and average together
        preds_list = []
        for i in range(len(saved_model_dirs)):
            preds_i = predict_from_saved_model(saved_model_dirs[i], mutants, DATA_MEANS[i], DATA_STDS[i], all_pams)
            preds_list.append(preds_i)
        preds_list_numbers_only = np.array([preds_list[n][all_pams] for n in range(len(preds_list))])
        preds_numbers = np.mean(preds_list_numbers_only, axis=0)
        preds = preds_list[0]
        preds[all_pams] = preds_numbers

        # perform selection
        round_output_variants, round_high_score = selection_function(rates_df=preds,
                                                                     n_best=n_best_variants_per_round,
                                                                     pam_to_max=pam_to_max,
                                                                     pam_to_min=pam_to_min,
                                                                     high_pams=high_pams,
                                                                     high_cutoffs=high_cutoffs,
                                                                     low_pams=low_pams,
                                                                     low_cutoffs=low_cutoffs,
                                                                     pams_to_max_selectivity=pams_to_max_selectivity,
                                                                     ramp_up_rounds=ramp_up_rounds,
                                                                     curr_round=curr_round,
                                                                     all_pams=all_pams)




        if (curr_high_score == None) or (round_high_score > curr_high_score):
            curr_high_score = round_high_score
        else:
            plateau_rounds += 1

        if plateau_rounds == decay_after_n_rounds_plateau:
            mutations_per_variant -= 1
            plateau_rounds = 0

        curr_best_df = pd.concat([curr_best_df, round_output_variants], axis=0, ignore_index=True)
        curr_best_df = curr_best_df.drop_duplicates()
        curr_best_df = curr_best_df.nlargest(n_best_variants_per_round, columns="sorted_on", keep="first")
        curr_best_df = curr_best_df.reset_index(drop=True)
        starting_variants = [list(curr_best_df[aa_positions].iloc[i, ]) for i in range(len(curr_best_df))]
        print("Current best variant: ", starting_variants[0])
        print("________________________________________________________________")
        trajectory.append(curr_best_df)

    return (curr_best_df, trajectory)



def main():
    args = parse_args()
    saved_model_dirs = args.saved_model_dirs or DEFAULT_SAVED_MODEL_DIRS

    output_dir = os.path.join(args.output_dir, "evolved", args.pred_run_name)
    ensure_dir(output_dir)

    with open(saved_model_dirs[0] + "/data_parameters.pkl", 'rb') as read_file:
        training_vars = pkl.load(read_file)
    run_name, model_name, three_nt_PAM, test_fraction, LOW_RATE_CONSTANT_CUTOFF, LOW_RATE_REPLACE, \
        HEATMAP_MAX, HEATMAP_MIN, nts, all_positions, pams = training_vars

    global DATA_MEANS, DATA_STDS
    DATA_MEANS = []
    DATA_STDS = []
    for directory in saved_model_dirs:
        with open(directory + "/mean_std.pkl", 'rb') as read_file:
            DATA_MEAN_i, DATA_STD_i = pkl.load(read_file)
            DATA_MEANS.append(DATA_MEAN_i)
            DATA_STDS.append(DATA_STD_i)

    starting_variants = prepare_starting_variants(args, all_positions)

    high_pams = args.high_pams
    high_cutoffs = args.high_cutoffs
    low_pams = args.low_pams
    low_cutoffs = args.low_cutoffs
    validate_cutoff_lists(high_pams, high_cutoffs, "High")
    validate_cutoff_lists(low_pams, low_cutoffs, "Low")

    pams_to_max_selectivity = normalize_pams_to_max_selectivity(args.pams_to_max_selectivity)
    if (args.pam_to_max is None) and (args.pam_to_min is None) and (pams_to_max_selectivity is None):
        pams_to_max_selectivity = DEFAULT_PAMS_TO_MAX_SELECTIVITY

    best_variants, traj = evolve(selection_function=select_custom,
                                 mutations_per_variant=args.mutations_per_variant,
                                 variants_per_round=args.variants_per_round,
                                 n_best_variants_per_round=args.n_best_variants_per_round,
                                 decay_after_n_rounds_plateau=args.decay_after_n_rounds_plateau,
                                 saved_model_dirs=saved_model_dirs,
                                 all_pams=pams,
                                 aa_positions=all_positions,
                                 pam_to_max=args.pam_to_max,
                                 pam_to_min=args.pam_to_min,
                                 high_pams=high_pams,
                                 high_cutoffs=high_cutoffs,
                                 low_pams=low_pams,
                                 low_cutoffs=low_cutoffs,
                                 pams_to_max_selectivity=pams_to_max_selectivity,
                                 starting_variants=starting_variants,
                                 ramp_up_rounds=args.ramp_up_rounds)

    final_csv = os.path.join(output_dir, "final_variants.csv")
    traj[-1].to_csv(final_csv)

    if args.save_traj:
        for i in range(len(traj)):
            traj[i].to_csv(os.path.join(output_dir, f"selection_round{i}.csv"))


if __name__ == "__main__":
    main()
