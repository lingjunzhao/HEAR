import json
import logging
from typing import List
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from transformers import AutoTokenizer, BertTokenizer


from airbert import Airbert, BERT_CONFIG_FACTORY

from utils.cli import get_parser
from utils.dataset.common import pad_packed
from utils.dataset.beam_dataset import BeamDataset
from utils.dataset.perturbate_dataset import PerturbateDataset
from utils.dataset.calibration_dataset import CalibrationDataset
from utils.dataset import PanoFeaturesReader
from utils.distributed import set_cuda, wrap_distributed_model, get_local_rank

from airbert import Airbert
from train import get_model_input, get_mask_options

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    # ----- #
    # setup #
    # ----- #

    # command line parsing
    parser = get_parser(training=False)
    args = parser.parse_args()

    # force arguments
    args.batch_size = 1

    print(args)

    # get device settings
    default_gpu, rank, device = set_cuda(args)

    # create output directory
    save_folder = os.path.join(args.output_dir, f"run-{args.save_name}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ------------ #
    # data loaders #
    # ------------ #

    # load a dataset
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    features_reader = PanoFeaturesReader(args.img_feature)

    dataset = CalibrationDataset(
        file_path=args.calibration_input_file,
        tokenizer=tokenizer,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        num_beams=args.num_beams,
        training=False,
        default_gpu=default_gpu,
        ground_truth_trajectory=False,
        highlighted_language=args.highlighted_language,
        shuffle_visual_features=False,
    )

    data_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- #
    # model #
    # ----- #

    config = BERT_CONFIG_FACTORY[args.model_name].from_json_file(args.config_file)
    config.cat_highlight = args.cat_highlight # type: ignore
    config.no_ranking = False # type: ignore
    config.masked_language = False # type: ignore
    config.masked_vision = False # type: ignore
    config.model_name = args.model_name
    model = Airbert.from_pretrained(args.from_pretrained, config, default_gpu=True)
    model.cuda()
    logger.info(f"number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---------- #
    # evaluation #
    # ---------- #

    with torch.no_grad():
        if args.test_select_threshold is False:
            all_scores = eval_epoch(model, data_loader, dataset, args)
        else:
            all_scores = eval_epoch_select_threshold(model, data_loader, dataset, args)

    # save scores
    file_name = os.path.basename(args.calibration_input_file)
    scores_path = os.path.join(save_folder, f"{args.prefix}_sigmoid_scores_{file_name}")
    json.dump(all_scores, open(scores_path, "w"), indent=2)
    logger.info(f"saving scores: {scores_path}")


def eval_epoch(model, data_loader, dataset, args):
    device = next(model.parameters()).device
    sigmoid = nn.Sigmoid()

    model.eval()
    # all_scores = []
    instrid2score, instrid2predicted = {}, {}
    counts, correct = 0, 0
    perturb_type2count, perturb_type2correct = defaultdict(int), defaultdict(int)
    for batch in tqdm(data_loader):
        # load batch on gpu
        batch = tuple(t.cuda(device=device, non_blocking=True) if hasattr(t, "cuda") else t for t in batch)
        instr_ids = get_instr_ids(batch)
        target = batch[0]
        target = target.detach().cpu().numpy()

        # get the model output
        output = model(*get_model_input(batch))
        opt_mask = get_mask_options(batch)
        vil_logit = pad_packed(output['ranking'].squeeze(1), opt_mask)
        sigmoid_outputs = sigmoid(vil_logit)
        predicted = sigmoid_outputs.round().detach().cpu().numpy()
        sigmoid_outputs = sigmoid_outputs.detach().cpu().numpy()
        num_candidates = get_num_candidates(batch).detach().cpu().numpy()

        for i, (instr_id, sigmoid_output) in enumerate(zip(instr_ids, sigmoid_outputs.tolist())):
            num_candidate = num_candidates[i]
            num_candidate = min(int(num_candidate), np.size(target[i]))
            valid_target = target[i][:num_candidate]
            valid_predicted = predicted[i][:num_candidate]
            valid_sigmoid_output = sigmoid_output[:num_candidate]
            instrid2score[instr_id] = valid_sigmoid_output
            instrid2predicted[instr_id] = valid_predicted
            counts += np.size(valid_target)
            correct += np.sum(valid_predicted == valid_target)

    accuracy = float(correct) / counts
    print("Accuracy: ", accuracy)

    result_json = {}
    dataset = dataset.get_dataset()
    result_key = "sigmoid_output"
    for item in dataset:
        item.pop("instruction_tokens", None)
        item.pop("instruction_token_masks", None)
        item.pop("instruction_segment_ids", None)
        instr_id = item["instr_id"]
        score = instrid2score[instr_id]
        item[result_key] = score
        result_json[instr_id] = item

        if "perturb_types" in item:
            predicted = instrid2predicted[instr_id]
            labels = item["labels"]
            perturb_start_idx = len(labels) // 2
            perturb_types = item["perturb_types"][perturb_start_idx:]
            for j, perturb_type in enumerate(perturb_types):
                perturb_type2count[perturb_type] += 2
                if labels[j + perturb_start_idx] == predicted[j + perturb_start_idx]:
                    perturb_type2correct[perturb_type] += 1
                if labels[j] == predicted[j]:
                    perturb_type2correct[perturb_type] += 1

    for perturb_type, perturb_count in perturb_type2count.items():
        perturb_correct = perturb_type2correct[perturb_type]
        perturb_accuracy = float(perturb_correct) / perturb_count
        print("Perturb type {} accuracy: {}".format(perturb_type, perturb_accuracy))

    return result_json


def eval_epoch_select_threshold(model, data_loader, dataset, args):
    device = next(model.parameters()).device
    sigmoid = nn.Sigmoid()

    model.eval()
    # all_scores = []
    instrid2score, instrid2predicted = {}, {}
    counts, correct = 0, 0
    perturb_type2count, perturb_type2correct = defaultdict(int), defaultdict(int)

    # select threshold based on best accuracy
    best_accuracy = 0.0
    best_threshold = 0.0
    for threshold in np.arange(0.1, 1.0, 0.1):
        for batch in tqdm(data_loader):
            # load batch on gpu
            batch = tuple(t.cuda(device=device, non_blocking=True) if hasattr(t, "cuda") else t for t in batch)
            instr_ids = get_instr_ids(batch)
            target = batch[0]
            target = target.detach().cpu().numpy()

            # get the model output
            output = model(*get_model_input(batch))
            opt_mask = get_mask_options(batch)
            vil_logit = pad_packed(output['ranking'].squeeze(1), opt_mask)
            sigmoid_outputs = sigmoid(vil_logit)
            # predicted = sigmoid_outputs.round().detach().cpu().numpy()
            predicted = (sigmoid_outputs > threshold).int().detach().cpu().numpy()
            sigmoid_outputs = sigmoid_outputs.detach().cpu().numpy()
            num_candidates = get_num_candidates(batch).detach().cpu().numpy()

            for i, (instr_id, sigmoid_output) in enumerate(zip(instr_ids, sigmoid_outputs.tolist())):
                num_candidate = num_candidates[i]
                num_candidate = min(int(num_candidate), np.size(target[i]))
                valid_target = target[i][:num_candidate]
                valid_predicted = predicted[i][:num_candidate]
                valid_sigmoid_output = sigmoid_output[:num_candidate]
                instrid2score[instr_id] = valid_sigmoid_output
                instrid2predicted[instr_id] = valid_predicted
                counts += np.size(valid_target)
                correct += np.sum(valid_predicted == valid_target)

        accuracy = float(correct) / counts
        print("Accuracy using threshold={}: {}".format(threshold, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print("Best accuracy: {} with threshold={}".format(best_accuracy, best_threshold))

    result_json = {}

    return result_json


def convert_scores(all_scores, beam_path, add_exploration_path=False):
    beam_data = json.load(open(beam_path, "r"))
    instr_id_to_beams = {item["instr_id"]: item["ranked_paths"] for item in beam_data}
    instr_id_to_exploration_path = {}
    if add_exploration_path:
        instr_id_to_exploration_path = {
            item["instr_id"]: item["exploration_path"] for item in beam_data
        }

    output = []
    for instr_id, scores in all_scores:
        idx = np.argmax(scores)
        beams = instr_id_to_beams[instr_id]
        trajectory = []
        if add_exploration_path:
            trajectory += instr_id_to_exploration_path[instr_id]
        # perturbations -> we fake a wrong destination by stopping at the initial location
        if idx >= len(beams):
            trajectory = [beams[0][0]]
        else:
            trajectory += beams[idx]
        output.append({"instr_id": instr_id, "trajectory": trajectory})

    return output


# ------------- #
# batch parsing #
# ------------- #


def get_instr_ids(batch) -> List[str]:
    instr_ids = batch[12]
    return instr_ids
    # return [str(item[0].item()) + "_" + str(item[1].item()) for item in instr_ids]


def get_num_candidates(batch):
    return batch[14]


if __name__ == "__main__":
    main()
