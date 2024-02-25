# pylint: disable=no-member, not-callable
import logging
import os
from typing import List, Iterator, TypeVar, Union, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import Dataset

from utils.dataset.common import (
    get_headings,
    get_viewpoints,
    load_distances,
    load_json_data,
    load_speaker_json_data,
    load_speaker_path_sampling_json_data,
    load_nav_graphs,
    randomize_regions,
    randomize_tokens,
    save_json_data,
    tokenize,
)
from utils.dataset.features_reader import FeaturesReader

logger = logging.getLogger(__name__)


class SpeakerDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        path_sampling: int,
        tokenizer: BertTokenizer,
        pano_features_reader: FeaturesReader,
        max_instruction_length: int,
        max_path_length: int,
        max_num_boxes: int,
        num_beams: int,
        num_beams_strict: bool,
        training: bool,
        masked_vision: bool,
        masked_language: bool,
        default_gpu: bool,
        **kwargs,
    ):
        # load and tokenize data (with caching)
        if path_sampling:
            self._vln_data = load_speaker_path_sampling_json_data(file_path, sample_size=path_sampling)
        else:
            self._vln_data = load_speaker_json_data(file_path)
        tokenize(self._vln_data, tokenizer, max_instruction_length, key="generated_instr")
        # save_json_data(self._vln_data, tokenized_path)
        self._tokenizer = tokenizer

        # load navigation graphs
        scan_list = [item["scan"] for item in self._vln_data]
        self._graphs = load_nav_graphs(scan_list)
        self._distances = load_distances(scan_list)

        # get all of the viewpoints for this dataset
        self._viewpoints = get_viewpoints(scan_list, self._graphs, pano_features_reader)

        # in training we only need 4 beams
        if training:
            num_beams = 4
            num_beams_strict = False

        self._num_beams = num_beams

        # # load beamsearch data
        # temp_beam_data = load_json_data(beam_path)
        #
        # # filter beams based on length
        # self._beam_data = []
        # for idx, item in enumerate(temp_beam_data):
        #     if len(item["ranked_paths"]) >= num_beams:
        #         if num_beams_strict:
        #             item["ranked_paths"] = item["ranked_paths"][:num_beams]
        #         self._beam_data.append(item)
        #     elif default_gpu:
        #         logger.warning(
        #             f"skipping index: {idx} in beam data in from path: {beam_path}"
        #         )

        # # get mapping from path id to vln index
        # path_to_vln = {}
        # for idx, vln_item in enumerate(self._vln_data):
        #     path_to_vln[vln_item["path_id"]] = idx
        #
        # # get mapping from beam to vln
        # self._beam_to_vln = {}
        # for idx, beam_item in enumerate(self._beam_data):
        #     path_id = int(beam_item["instr_id"].split("_")[0])
        #     self._beam_to_vln[idx] = path_to_vln[path_id]

        self._pano_features_reader = pano_features_reader
        self._max_instruction_length = max_instruction_length
        self._max_path_length = max_path_length
        self._max_num_boxes = max_num_boxes
        self._training = training
        self._masked_vision = masked_vision
        self._masked_language = masked_language
        self._path_sampling = path_sampling

    def __len__(self):
        return len(self._vln_data)

    def get_dataset(self):
        return self._vln_data

    def __getitem__(self, vln_index):
        # vln_index = self._beam_to_vln[beam_index]

        # get beam info
        instr_id = self._vln_data[vln_index]["instr_id"]
        path_id, instruction_index = instr_id.split("_", 1)
        path_id = int(path_id)

        # get vln info
        scan_id = self._vln_data[vln_index]["scan"]
        heading = self._vln_data[vln_index]["heading"]
        if self._path_sampling:
            gt_path = self._vln_data[vln_index]["pred_path"]
        else:
            gt_path = self._vln_data[vln_index]["path"]

        # get the instruction data
        instr_tokens = self._vln_data[vln_index]["instruction_tokens"][
            0
        ]
        instr_masks = self._vln_data[vln_index]["instruction_token_masks"][
            0
        ]
        segment_ids = self._vln_data[vln_index]["instruction_segment_ids"][
            0
        ]
        instr_highlights = torch.zeros((self._num_beams, 0)).long()

        # get all of the paths
        beam_paths = []
        beam_paths.append(gt_path)
        # for ranked_path in self._beam_data[beam_index]["ranked_paths"]:
        #     beam_paths.append([p for p, _, _ in ranked_path])

        success = self._get_path_success(scan_id, gt_path, beam_paths)
        # if self._training:
        #     # select one positive and three negative paths
        #     if np.sum(success == 1) == 0 or np.sum(success == 0) < 3:
        #         # not enough positive or negative paths (this should be rare)
        #         target = -1  # default ignore index
        #         selected_paths = beam_paths[:4]
        #     else:
        #         target = 0
        #         selected_paths = []
        #         # first select a positive
        #         idx = np.random.choice(np.where(success == 1)[0])
        #         selected_paths.append(beam_paths[idx])
        #         # next select three negatives
        #         idxs = np.random.choice(np.where(success == 0)[0], size=3)
        #         for idx in idxs:
        #             selected_paths.append(beam_paths[idx])
        # else:
        target = success
        selected_paths = beam_paths

        # get path features
        features, boxes, probs, masks = [], [], [], []
        for path in selected_paths:
            f, b, p, m = self._get_path_features(scan_id, path, heading)
            features.append(f)
            boxes.append(b)
            probs.append(p)
            masks.append(m)

        # convert data into tensors
        image_features = torch.tensor(features).float()
        image_boxes = torch.tensor(boxes).float()
        image_probs = torch.tensor(probs).float()
        image_masks = torch.tensor(masks).long()
        instr_tokens = torch.tensor([instr_tokens] * len(features)).long()
        instr_masks = torch.tensor([instr_masks] * len(features)).long()
        segment_ids = torch.tensor([segment_ids] * len(features)).long()
        instr_highlights = torch.zeros((len(features), 0)).long()

        # randomly mask image features
        if self._masked_vision:
            image_features, image_targets, image_targets_mask = randomize_regions(
                image_features, image_probs, image_masks
            )
        else:
            image_targets = torch.ones_like(image_probs) / image_probs.shape[-1]
            image_targets_mask = torch.zeros_like(image_masks)

        # randomly mask instruction tokens
        if self._masked_language:
            instr_tokens, instr_targets = randomize_tokens(
                instr_tokens, instr_masks, self._tokenizer
            )
        else:
            instr_targets = torch.ones_like(instr_tokens) * -1

        # set target
        target = torch.tensor(target).long()

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self._max_path_length * self._max_num_boxes, self._max_instruction_length
        ).long()
        # instr_id = torch.tensor([path_id, instruction_index])

        num_candidates = torch.tensor(1).long()

        return (
            target,
            image_features,
            image_boxes,
            image_masks,
            image_targets,
            image_targets_mask,
            instr_tokens,
            instr_masks,
            instr_targets,
            instr_highlights,
            segment_ids,
            co_attention_mask,
            instr_id,
            torch.ones(image_features.shape[0]).bool(),
            num_candidates,
        )

    def _get_path_success(self, scan_id, path, beam_paths, success_criteria=3):
        d = self._distances[scan_id]
        success = np.zeros(len(beam_paths))
        for idx, beam_path in enumerate(beam_paths):
            if d[path[-1]][beam_path[-1]] < success_criteria:
                success[idx] = 1
        return success

    # TODO move to utils
    def _get_path_features(self, scan_id: str, path: List[str], first_heading: float):
        """ Get features for a given path. """
        headings = get_headings(self._graphs[scan_id], path, first_heading)
        # for next headings duplicate the last
        next_headings = headings[1:] + [headings[-1]]

        path_length = min(len(path), self._max_path_length)
        path_features, path_boxes, path_probs, path_masks = [], [], [], []
        for path_idx, path_id in enumerate(path[:path_length]):
            key = scan_id + "-" + path_id

            # get image features
            features, boxes, probs = self._pano_features_reader[
                key, headings[path_idx], next_headings[path_idx],
            ]
            num_boxes = min(len(boxes), self._max_num_boxes)

            # pad features and boxes (if needed)
            pad_features = np.zeros((self._max_num_boxes, 2048))
            pad_features[:num_boxes] = features[:num_boxes]

            pad_boxes = np.zeros((self._max_num_boxes, 12))
            pad_boxes[:num_boxes, :11] = boxes[:num_boxes, :11]
            pad_boxes[:, 11] = np.ones(self._max_num_boxes) * path_idx

            pad_probs = np.zeros((self._max_num_boxes, 1601))
            pad_probs[:num_boxes] = probs[:num_boxes]

            box_pad_length = self._max_num_boxes - num_boxes
            pad_masks = [1] * num_boxes + [0] * box_pad_length

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        # pad path lists (if needed)
        for path_idx in range(path_length, self._max_path_length):
            pad_features = np.zeros((self._max_num_boxes, 2048))
            pad_boxes = np.zeros((self._max_num_boxes, 12))
            pad_boxes[:, 11] = np.ones(self._max_num_boxes) * path_idx
            pad_probs = np.zeros((self._max_num_boxes, 1601))
            pad_masks = [0] * self._max_num_boxes

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        return (
            np.vstack(path_features),
            np.vstack(path_boxes),
            np.vstack(path_probs),
            np.hstack(path_masks),
        )