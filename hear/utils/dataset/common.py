import json
from typing import List, Dict, Union, TypeVar, Iterator
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from transformers import PreTrainedTokenizer


def pad_packed(t: torch.Tensor, mask: Union[torch.Tensor, torch.BoolTensor]) -> torch.Tensor:
    mask = mask.bool()
    out = mask.clone().to(t.dtype)
    out[mask] = t
    out[~mask] = -float("inf")
    return out


def load_json_data(path):
    with open(path, "r") as fid:
        data = json.load(fid)
    return data


def load_speaker_json_data(path):
    data = []
    count = 0
    with open(path, "r") as fid:
        tmp_data = json.load(fid)
        for instr_id, item in tmp_data.items():
            if "generated_instrs" in item and not item["generated_instrs"]:
                print("Skipping emtpy instr_id: ", instr_id)
                continue
            data.append(item)
            count += 1

    print("Loaded {} data from {}".format(count, path))

    return data


def load_speaker_path_sampling_json_data(path, sample_size=10):
    data = []
    path_sample_prefix = "pred_path_sample_"
    result_sample_prefix = "result_sample_"

    with open(path, "r") as fid:
        tmp_data = json.load(fid)
        for instr_id, item in tmp_data.items():
            base_item = {x: item[x] for x in item if (not x.startswith(path_sample_prefix) and not x.startswith(result_sample_prefix))}
            for i in range(sample_size):
                new_item = copy.deepcopy(base_item)
                new_item["instr_id"] = new_item["instr_id"] + "_sample_" + str(i)
                new_item["pred_path"] = item[path_sample_prefix + str(i)]
                new_item["result"] = item[result_sample_prefix + str(i)]
                data.append(new_item)

    return data


def save_json_data(data, path):
    with open(path, "w") as fid:
        json.dump(data, fid, indent=2)


def load_nav_graphs(scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    graphs = {}
    for scan in scans:
        with open("data/connectivity/%s_connectivity.json" % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item["included"]:
                    for j, conn in enumerate(item["unobstructed"]):
                        if conn and data[j]["included"]:
                            positions[item["image_id"]] = np.array(
                                [item["pose"][3], item["pose"][7], item["pose"][11]]
                            )
                            assert data[j]["unobstructed"][
                                i
                            ], "Graph should be undirected"
                            G.add_edge(
                                item["image_id"],
                                data[j]["image_id"],
                                weight=distance(item, data[j]),
                            )
            nx.set_node_attributes(G, values=positions, name="position")
            graphs[scan] = G
    return graphs


def load_distances(scans):
    distances = {}
    for scan in scans:
        with open(f"data/distances/{scan}_distances.json", "r") as fid:
            distances[scan] = json.load(fid)
    return distances


def get_headings(g, path, first_heading):
    # get xy positions for path
    pos = nx.get_node_attributes(g, "position")
    pos = {node: pos[node][:2] for node in path}

    # calculate headdings
    headings = [first_heading]
    for source, target in zip(path[:-1], path[1:]):
        dx = pos[target][0] - pos[source][0]
        dy = pos[target][1] - pos[source][1]
        # use dx/dy because heading is from north (i.e. y)
        headings.append(np.arctan2(dx, dy))
    return headings


def index(subseq, seq):
    i, n, m = -1, len(seq), len(subseq)
    try:
        while True:
            i = seq.index(subseq[0], i + 1, n - m + 1)
            if subseq == seq[i : i + m]:
                return i
    except ValueError:
        return -1


def tokenize(
    data: List[Dict], tokenizer: PreTrainedTokenizer, max_instruction_length: int, key="instructions"
):
    pad, cls, sep = tokenizer.convert_tokens_to_ids(["[PAD]", "[CLS]", "[SEP]"])  # type: ignore
    highlight_tokens = ["[unused0]", "[unused1]", "[unused2]"]
    begin_hl, end_hl, remove_id = tokenizer.convert_tokens_to_ids(highlight_tokens)
    print("highlight token ids: ", begin_hl, end_hl)
    print("remove token id: ", remove_id)
    tokenizer.add_special_tokens({'additional_special_tokens': highlight_tokens})

    for item in data:
        item["instruction_tokens"] = []
        item["instruction_token_masks"] = []
        item["instruction_segment_ids"] = []
        if "highlights" in item:
            item["instruction_highlights"] = []
        if "perturbations" in item:
            item["perturbation_tokens"] = [[] for _ in item["instructions"]]
        if "perturbation_highlights" in item:
            item["perturbation_highlight_masks"] = [[] for _ in item["instructions"]]

        if key == "instructions" or key == "generated_instrs":
            instructions = item[key]
            if len(instructions) == 0:
                print("Skipping emtpy instr_id: ", item["instr_id"])
                continue
        elif key == "generated_instr":
            instructions = [item[key]]
        else:
            print("Unknown item key: ", key)
            print(item)
            continue

        for i, instruction in enumerate(instructions):
            # print(instruction)
            tokens = tokenizer.tokenize(instruction)
            # print("tokenized: ")
            # print(tokens)

            # add a classification and seperator tokens
            # tokens = [cls] + [
            #     tokenizer.vocab[token] for token in tokens  # type: ignore
            # ]
            tokens = [cls] + tokenizer.convert_tokens_to_ids(tokens)  # type: ignore
            tokens = tokens[: max_instruction_length - 1] + [sep]
            masks = [1] * len(tokens)
            segment_ids = [0] * len(tokens)
            # print(tokens)

            pad_length = max_instruction_length - len(tokens)
            pad_tokens = tokens + [pad] * pad_length
            masks = masks + [0] * pad_length
            segment_ids = segment_ids + [0] * pad_length

            item["instruction_tokens"].append(pad_tokens)
            item["instruction_token_masks"].append(masks)
            item["instruction_segment_ids"].append(segment_ids)

            # create a highlight version
            if "highlights" in item:
                highlights = []
                cursor = 0
                for word in item["highlights"][i]:
                    token = tokenizer.tokenize(word)
                    token_id = [tokenizer.vocab[t] for t in token]  # type: ignore
                    increment = index(token_id, tokens[cursor:])
                    if increment == -1:
                        continue
                    highlights += [False] * increment + [True] * len(token_id)
                    cursor += increment + len(token_id)

                # pad lists
                pad_length = max_instruction_length - len(highlights)
                highlights = highlights + [False] * pad_length

                # add to data
                item["instruction_highlights"].append(highlights)

            # create a perturbation version
            if "perturbations" in item:
                for j, inst in enumerate(item["perturbations"][i]):
                    tokens = tokenizer.tokenize(inst)
                    tokens = [cls] + [
                        tokenizer.vocab[token] for token in tokens  # type: ignore
                    ]
                    tokens = tokens[: max_instruction_length - 1] + [sep]
                    pad_length = max_instruction_length - len(tokens)
                    pad_tokens = tokens + [pad] * pad_length
                    item["perturbation_tokens"][i].append(pad_tokens)

                    # create a perturbation + highlight version
                    if "perturbation_highlights" in item:
                        highlights = []
                        cursor = 0
                        for word in item["perturbation_highlights"][i][j]:
                            token = tokenizer.tokenize(word)
                            token_id = [tokenizer.vocab[t] for t in token]  # type: ignore
                            increment = index(token_id, tokens[cursor:])
                            if increment == -1:
                                continue
                            highlights += [False] * increment + [True] * len(token_id)
                            cursor += increment + len(token_id)

                        # pad lists
                        pad_length = max_instruction_length - len(highlights)
                        highlights = highlights + [False] * pad_length

                        # add to data
                        item["perturbation_highlight_masks"][i].append(highlights)


def load_tokens(
    path: Union[Path, str], tokenizer: PreTrainedTokenizer, max_instruction_length: int
) -> List[Dict]:
    ppath = Path(path)
    assert ppath.suffix == ".json", ppath

    # load and tokenize data (with caching)
    tokenized_path = (
        ppath.parent / f"{ppath.stem}_tokenized_{max_instruction_length}{ppath.suffix}"
    )

    if tokenized_path.is_file():
        data = load_json_data(tokenized_path)
    else:
        data = load_json_data(ppath)
        tokenize(data, tokenizer, max_instruction_length)
        save_json_data(data, tokenized_path)
    return data


def randomize_tokens(tokens, mask, tokenizer):
    """ Return tokens randomly masked using standard BERT probabilities. """
    targets = torch.ones_like(tokens) * -1

    # get random data
    p = torch.rand_like(tokens.float()) * mask.float()
    random_tokens = torch.randint_like(tokens, len(tokenizer.vocab))

    # set targets for masked tokens
    thresh = 0.85
    targets[p >= thresh] = tokens[p >= thresh]

    # progressively overwrite tokens while increasing the threshold

    # replace 80% with '[MASK]' token
    tokens[p >= thresh] = tokenizer.vocab["[MASK]"]

    # replace 10% with a random word
    thresh = 0.85 + 0.15 * 0.8
    tokens[p >= thresh] = random_tokens[p >= thresh]

    # keep 10% unchanged
    thresh = 0.85 + 0.15 * 0.9
    tokens[p >= thresh] = targets[p >= thresh]

    return tokens, targets


def randomize_regions(features, probs, mask):
    """Return features after randomly masking using ViLBERT probabilities.

    Let B equal the batch size and N equal the number of regions.

    Parameters
    ----------
    features : torch.tensor, (B, N, 2048)
        The original feature vectors.
    probs : torch.tensor, (B, N, 2048)
        The target probability distribution for each region.
    mask : torch.tensor, (B, N)
        A zero-one mask where zeros represent missing regions.
    """
    targets = torch.ones_like(probs) / probs.shape[-1]
    targets_mask = torch.zeros_like(mask)

    p = torch.rand_like(mask.float()) * mask.float()

    # set targets for masked regions
    thresh = 0.85
    targets[p >= thresh] = probs[p >= thresh]
    targets_mask[p >= thresh] = 1

    # replace 90% of the masked features with zeros
    thresh = 0.85 + 0.15 * 0.1
    features[p >= thresh] = 0

    return features, targets, targets_mask


def get_viewpoints(scan_list, graphs, feature_reader):
    """ Return a list of viewpoints that are in the graphs and feature reader. """
    viewpoints = {}
    for scan in scan_list:
        graph_viewpoints = set(graphs[scan].nodes())
        feats_viewpoints = feature_reader.viewpoints[scan]
        viewpoints[scan] = feats_viewpoints.intersection(graph_viewpoints)
    return viewpoints

T = TypeVar("T")


def shuffle_different(seq: List[T]) -> Iterator[List[T]]:
    sequences = list(itertools.permutations(seq, len(seq)))
    random.shuffle(sequences)
    for s in sequences:
        l = list(s)
        if l != seq:
            yield l


def shuffle_non_adjacent(seq: List[T]) -> Iterator[List[T]]:
    n = len(seq)
    starting = {i: [j for j in range(n) if abs(j - i) > 1] for i in range(n)}
    keys = list(starting.keys())
    done = []
    while keys != []:
        idx_keys, start = random.choice(list(enumerate(keys)))
        idx_list, permute = random.choice(list(enumerate(starting[start])))

        del starting[start][idx_list]
        if starting[start] == []:
            del keys[idx_keys]

        if {start, permute} in done:
            continue
        done.append({start, permute})

        shuffled = copy.deepcopy(seq)
        shuffled[start], shuffled[permute] = shuffled[permute], shuffled[start]

        yield shuffled
