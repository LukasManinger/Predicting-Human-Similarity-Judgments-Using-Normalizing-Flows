import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import csv
import datetime
from collections import namedtuple
from typing import Callable, List, Union

import numpy as np
import psiz
from numpy import typing as npt
from tensorflow import keras
from tqdm import tqdm

from src import ROOT_DIR
from image_manager import ImageManager
from predictors import (
    GlowBayesPredictor,
    GlowHalfPredictor,
    GlowPredictor,
    GlowSimplePredictor,
    GlowWeightedPredictor,
    PcaPredictor,
    RandomPredictor,
    ResnetPredictor,
    SimplePredictor,
)


def main() -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(f"{now = }")

    cmd_args = arg_parsing()
    print(f"{cmd_args = }")

    obs, catalog = psiz.datasets.load_dataset("birds-16", verbose=1)
    # print(obs, type(obs))
    # print(catalog, type(catalog))

    image_manager = ImageManager(catalog, cmd_args.disable_tqdm)

    dist: Callable = None
    if cmd_args.dist == "mse":
        dist = keras.losses.MeanSquaredError()
    elif cmd_args.dist == "cos":
        dist = keras.losses.CosineSimilarity()
    assert dist is not None

    if cmd_args.predictor == "simple":
        predictor = SimplePredictor(image_manager, dist, cmd_args.disable_tqdm)

    elif cmd_args.predictor == "random":
        predictor = RandomPredictor(image_manager, dist, cmd_args.disable_tqdm)

    elif cmd_args.predictor == "glow_simple":
        assert cmd_args.hps_path is not None
        predictor = GlowSimplePredictor(
            image_manager, dist, cmd_args.hps_path, cmd_args.disable_tqdm
        )

    elif cmd_args.predictor == "glow_bayes":
        assert cmd_args.hps_path is not None
        assert cmd_args.data_sigma is not None
        predictor = GlowBayesPredictor(
            image_manager, dist, cmd_args.hps_path, cmd_args.data_sigma, cmd_args.disable_tqdm
        )

    elif cmd_args.predictor == "glow_weighted":
        assert cmd_args.hps_path is not None
        predictor = GlowWeightedPredictor(
            image_manager, dist, cmd_args.hps_path, cmd_args.disable_tqdm
        )

    elif cmd_args.predictor == "glow_half":
        assert cmd_args.hps_path is not None
        predictor = GlowHalfPredictor(image_manager, dist, cmd_args.hps_path, cmd_args.disable_tqdm)

    elif cmd_args.predictor == "pca":
        assert cmd_args.emb_name is not None
        predictor = PcaPredictor(image_manager, dist, cmd_args.emb_name, cmd_args.disable_tqdm)

    elif cmd_args.predictor == "resnet":
        assert cmd_args.hps_path is not None
        assert cmd_args.layer_to_use is not None
        predictor = ResnetPredictor(
            image_manager, dist, cmd_args.hps_path, cmd_args.layer_to_use, cmd_args.disable_tqdm
        )
    else:
        raise ValueError("Invalid argument: 'predictor'!")

    # results = []
    results_8c2 = []
    results_2c1 = []

    for i, (stimulus, n_reference, n_select) in enumerate(
        tqdm(
            zip(obs.stimulus_set, obs.n_reference, obs.n_select),
            desc="Predicting stimuli",
            total=obs.n_trial,
            colour="#d95f02",
            disable=cmd_args.disable_tqdm,
        )
    ):
        prediction = predictor.predict(stimulus)

        trial_type = f"{n_reference}choose{n_select}"

        # result = evaluate_prediction(trial_type, stimulus, prediction)
        result = triplet_accuracy(trial_type, stimulus, prediction)

        # print(" ----- ")
        # print(stimulus)
        # # print(prediction)
        # print([p.id for p in prediction])
        # print(result)

        if trial_type == "8choose2":
            results_8c2.append(result)
        else:
            assert trial_type == "2choose1"
            results_2c1.append(result)

        # if i > 10:
        #     break

        if cmd_args.disable_tqdm and i % 1000 == 0:
            print(f"Predicting stimuli: {i} / {obs.n_trial}")

    infix = ""
    if isinstance(predictor, GlowPredictor):
        infix = predictor.hps["NAME"]
    elif isinstance(predictor, PcaPredictor):
        infix = cmd_args.emb_name
    elif isinstance(predictor, ResnetPredictor):
        infix = f"{predictor.hps['ID'].split('_2022')[0]}_{cmd_args.layer_to_use}"  # HACK

    save_results(
        results_8c2,
        results_2c1,
        f"{predictor.__class__.__name__}_{cmd_args.dist}_{infix}_{now}",
        header=[str(cmd_args)],
    )

    print(np.mean(results_8c2), np.mean(results_2c1))

    # plot_percentages(
    #    [np.mean(results_8c2 + results_2c1), np.mean(results_8c2), np.mean(results_2c1)]
    # )


def arg_parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("predictor", metavar="PREDICTOR", type=str)

    parser.add_argument("--dist", type=str, default="mse")
    parser.add_argument("--disable_tqdm", type=int, default=0)
    parser.add_argument("--hps_path", type=str)
    parser.add_argument("--data_sigma", type=float, default=1.0)
    parser.add_argument("--emb_name", type=str, default="pca_emb_222")
    parser.add_argument("--layer_to_use", type=int, default=-1)

    return parser.parse_args()


def evaluate_prediction(
    trial_type: str, ranked_observation: npt.NDArray[np.int_], prediction
) -> float:
    print("Deprecation warning!")

    # Triplet accuracy (or not)
    if trial_type == "8choose2":
        # TODO Check if this is right
        correct = 0
        if ranked_observation[1] == prediction[1].id:
            correct += 7
        elif ranked_observation[1] == prediction[2].id:
            correct += 6

        if ranked_observation[2] == prediction[2].id:
            if ranked_observation[1] == prediction[1].id:
                correct += 6
            else:
                correct += 5
        elif ranked_observation[2] == prediction[1].id:
            correct += 6

        return correct / 13

    elif trial_type == "2choose1":
        return float(ranked_observation[1] == prediction[1].id)

    else:
        raise ValueError("Invalid trial_type!")


def triplet_accuracy(
    trial_type: str, ranked_observation: npt.NDArray[np.int_], prediction
) -> Union[int, List[int]]:
    # Apparently this is equivalent to the upper evaluation method
    if trial_type == "8choose2":
        pred_ids = np.array([p.id for p in prediction])
        # counter = 0
        results = []

        # First choice
        a = ranked_observation[1]
        a_pred_idx = np.asarray(pred_ids == a).nonzero()[0][0]
        for i in range(2, 9):
            # if a_pred_idx > 2:
            #    break

            other = ranked_observation[i]
            # counter += a_pred_idx < np.asarray(pred_ids == other).nonzero()[0][0]
            results.append(int(a_pred_idx < np.asarray(pred_ids == other).nonzero()[0][0]))

        # Second choice
        b = ranked_observation[2]
        b_pred_idx = np.asarray(pred_ids == b).nonzero()[0][0]
        for i in range(3, 9):
            # if b_pred_idx > 2:
            #    break

            other = ranked_observation[i]
            # counter += b_pred_idx < np.asarray(pred_ids == other).nonzero()[0][0]
            results.append(int(b_pred_idx < np.asarray(pred_ids == other).nonzero()[0][0]))

        # return counter / 13
        return results

    elif trial_type == "2choose1":
        return int(ranked_observation[1] == prediction[1].id)

    else:
        raise ValueError("Invalid trial_type!")


def save_results(results_8c2, results_2c1, file_name, header: List[str] = [""]):
    with open(f"{ROOT_DIR}/predicting/out/{file_name}.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(header)
        csv_writer.writerow(["trial_type", "result"])

        for result in results_8c2:
            csv_writer.writerow(["8choose2", result])

        for result in results_2c1:
            csv_writer.writerow(["2choose1", result])


def calc_chance_level():
    # Just testing
    ReferenceSimilarity = namedtuple("ReferenceSimilarity", ["id", "similarity"])

    obs, catalog = psiz.datasets.load_dataset("birds-16", verbose=1)

    # correct = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    correct = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7])

    result = []
    result_alt = []

    print(len(obs.outcome_idx_list[0]))

    # Wrong: outcomes are outcomes for trials not responses
    for outcome in obs.outcome_idx_list[0]:
        print(outcome)

        outcome = [ReferenceSimilarity(out, 0.0) for out in outcome]
        outcome.insert(0, ReferenceSimilarity(-1, 0.0))
        result.append(evaluate_prediction("8choose2", correct, outcome))
        result_alt.append(triplet_accuracy("8choose2", correct, outcome))

    print("chance level:", np.mean(result))
    print("chance level:", np.mean(result_alt))

    count_8c2 = 0
    count_2c1 = 0

    for n_reference in obs.n_reference:
        if n_reference == 8:
            count_8c2 += 1
        elif n_reference == 2:
            count_2c1 += 1

    print("8c2:", count_8c2)
    print("2c1:", count_2c1)


if __name__ == "__main__":
    main()
    # calc_chance_level()
