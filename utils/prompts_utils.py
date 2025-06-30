import os
import random
import numpy as np
from utils.evaluate_utils import contains_replacement_character

summary_template = "Sentences:\n{{Document}}"
simulation_template = "Concept: {{Concept}}\nActivations:\n<start>\n{{Document}}<end>"


def write_prompts(contents, concept_attention, cases_ids, tokenizer, vis_threshold, max_len):
    cnt = 1
    concept_prompt = ""
    for i in cases_ids:
        tokens = tokenizer.tokenize(contents[i])
        clean_tokens = [tokenizer.convert_tokens_to_string(token) for token in tokens]

        concept_prompt += 'Sentence ' + str(cnt) + ':\n<start>\n'
        for j in range(min(len(clean_tokens), max_len - 1)):
            if contains_replacement_character(clean_tokens[j]):
                continue
            if concept_attention[i][j + 1] > vis_threshold:
                concept_prompt += str(clean_tokens[j]) + '\t' + str(concept_attention[i][j + 1]) + '\n'
            else:
                concept_prompt += str(clean_tokens[j]) + '\t0\n'
        concept_prompt += '<end>\n'
        cnt += 1
    return concept_prompt


def write_summary_prompts(concept_ids, contents, slot_attention, tokenizer, prompt_path, topk=10, vis_threshold=0.3,
                          max_len=256, prefix="summarization_concept_"):
    concept_logits = np.sum(slot_attention, axis=-1)
    cases_ids = {}
    for concept_id in concept_ids:
        filtered_lst = [(i, x) for i, x in enumerate(concept_logits[:, concept_id])]
        indexed_lst = sorted(filtered_lst, key=lambda x: x[1], reverse=True)
        cases_ids[concept_id] = [item[0] for item in indexed_lst][:topk]

    for concept_id in concept_ids:
        concept_attention = slot_attention[:, concept_id, :]
        concept_prompt = write_prompts(contents, concept_attention, cases_ids[concept_id], tokenizer, vis_threshold, max_len)

        cur_prompt = summary_template.replace('{{Document}}', concept_prompt)
        file_name = prompt_path + prefix + str(concept_id) + '.txt'
        with open(file_name, 'w') as file:
            file.write(cur_prompt)


def write_simulation_prompts(concept_ids, contents, slot_attention, tokenizer, prompt_path, concept_summaries, topk=100,
                             max_len=256):
    if not os.path.exists(prompt_path):
        os.makedirs(prompt_path)

    for concept_id in concept_ids:
        file_name = prompt_path + '/concept' + str(concept_id)
        if not os.path.exists(file_name):
            os.makedirs(file_name)

        concept_logits = np.sum(slot_attention, axis=-1)[:, concept_id]

        cases_ids = sorted(range(len(concept_logits)), key=lambda i: concept_logits[i], reverse=True)
        all_cases = cases_ids[10: 10 + topk]

        for i in all_cases:
            tokens = tokenizer.tokenize(contents[i])
            clean_tokens = [tokenizer.convert_tokens_to_string(token) for token in tokens]

            content_prompt = ''
            for j in range(min(len(clean_tokens), max_len - 1)):
                if contains_replacement_character(clean_tokens[j]):
                    continue
                content_prompt += str(clean_tokens[j]) + '\n'

            cur_prompt = simulation_template.replace('{{Concept}}', concept_summaries[concept_id])

            with open(file_name + '/case' + str(i) + '.txt', 'w') as file:
                file.write(cur_prompt.replace('{{Document}}', content_prompt))
