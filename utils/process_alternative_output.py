import json
import spacy
import numpy as np


def process_calibration_output(input_json_file, instr_key="generated_instrs", prob_key="sigmoid_output", alternative_threshold=0.5):
    count_instr = 0
    highlight_start_tag, highlight_end_tag = "[unused0] ", "[unused1]"
    highlight_start_tag1 = "[unused0]"
    output_json = {}
    with open(input_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            if "is_alternative" in item:
                original_instr_id = item["original_instr_id"]
                if "id2alters" not in output_json[original_instr_id]:
                    output_json[original_instr_id]["id2alters"] = {}

                token_prob_list = []
                for prob, alternative_instr in zip(item[prob_key], item[instr_key]):
                    tokenized_alternative_instr = alternative_instr.split(" ")
                    start, end = tokenized_alternative_instr.index(highlight_start_tag1), tokenized_alternative_instr.index(highlight_end_tag)
                    alternative_token = " ".join(tokenized_alternative_instr[start+1:end])
                    if alternative_token == "[unused2]":
                        alternative_token = "REMOVE"
                    token_prob_list.append((alternative_token, prob))
                alternative_index = item["token_idx"]

                # get id2alters
                filtered_token_prob_list = [x for x in token_prob_list if x[1] >= alternative_threshold]
                output_json[original_instr_id]["id2alters"][alternative_index] = filtered_token_prob_list

            else:
                count_instr += 1
                instructions = item[instr_key]
                probs = item[prob_key]
                if len(instructions) != len(probs):
                    print("{} num generated instructions and probs not match! {} vs {}".format(instr_id, len(instructions), len(probs)))
                    continue

                output_json[instr_id] = item

    print("Number of instrs counted: ", count_instr)
    return output_json


def merge_confidence_phrase_output(hallucination_json, type_json_file, output_json_file, output_best_alter_key="best_alternative",
                            output_id2alter_key="id2alters", alternative_threshold=0.0, top_k=5, success_loc_file=None):
    output_json = {}
    nlp = spacy.load("en_core_web_sm")

    with open(type_json_file) as f:
        type_json = json.load(f)

    if success_loc_file is not None:
        with open(success_loc_file) as f:
            success_loc_json = json.load(f)

    for instr_id, hallucination_item in hallucination_json.items():
        type_item = type_json[instr_id]
        alternative_tokens = []
        probs = []
        candidate_indices = hallucination_item["candidate_indices"]
        id2intrinsic_alters = hallucination_item[output_id2alter_key]
        id2alters = {}
        id2prob = {}
        for i, hallucination_prob in enumerate(hallucination_item["sigmoid_output"]):
            probs.append(hallucination_prob)
            phrase_id = candidate_indices[i]
            type_prob_id = type_item["candidate_indices"].index(phrase_id)
            intrinsic_prob = type_item["sigmoid_output"][type_prob_id]

            new_alter_prob_list = []
            for alter, alter_prob in id2intrinsic_alters[phrase_id]:
                new_alter_prob = intrinsic_prob * alter_prob
                if new_alter_prob >= alternative_threshold:
                    new_alter_prob_list.append((alter, new_alter_prob))
            extrinsic_prob = 1 - intrinsic_prob
            new_alter_prob_list.append(("REMOVE", extrinsic_prob))
            new_alter_prob_list.sort(key=lambda x: x[1], reverse=True)
            new_alter_prob_list = new_alter_prob_list[:top_k]  # keep top k
            id2alters[phrase_id] = new_alter_prob_list
            alternative_tokens.append(new_alter_prob_list[0][0])

        # write prob and alternative to string
        doc = nlp(type_item["original_instr"])
        tokenized_instruction = []
        numbered_instruction = []
        for j, token in enumerate(doc):
            tokenized_instruction.append(token.text)
            numbered_instruction += [token.text, "({})".format(j)]

        idx2best_alternative = {}
        for i, candidate_indice in enumerate(candidate_indices):
            prob = probs[i]
            best_alternative = alternative_tokens[i]
            idx2best_alternative[candidate_indice] = best_alternative
            id2prob[candidate_indice] = prob

        type_item[output_best_alter_key] = idx2best_alternative
        type_item[output_id2alter_key] = id2alters
        type_item.pop("labels", None)
        type_item.pop("generated_instrs", None)
        type_item["intrinsic_prob"] = type_item["sigmoid_output"]
        type_item["sigmoid_output"] = id2prob
        type_item["original_instr"] = " ".join(tokenized_instruction)
        type_item["numbered_instruction"] = " ".join(numbered_instruction)
        if success_loc_file is not None:
            type_item["success_locations"] = success_loc_json[instr_id]["success_locations"]
        output_json[instr_id] = type_item

    with open(output_json_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    print('Saved file to %s' % output_json_file)


success_loc_file = "cal_data/speaker_t5_clip_greedy_val_seen_success_loc.json"
hallucination_type_json_file = "data/runs/run-test_hallucination_type/_sigmoid_scores_t5_val_seen_highlighted_phrase_gpt4_direction_dev_test.json"
input_json_file = "data/runs/run-test_hallucination_detection/_sigmoid_scores_t5_val_seen_highlighted_phrase_alters_gpt4_direction_dev_test.json"
output_json_file = "data/runs/run-test_hallucination_detection/_sigmoid_scores_t5_val_seen_highlighted_phrase_alters_gpt4_direction_dev_test_merged.json"

input_json = process_calibration_output(input_json_file, alternative_threshold=0.0)
merge_confidence_phrase_output(input_json, hallucination_type_json_file, output_json_file, alternative_threshold=0.0,
                               top_k=3, success_loc_file=success_loc_file)

