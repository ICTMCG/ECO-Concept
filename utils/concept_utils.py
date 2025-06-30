import json
import numpy as np
from utils.prompts_utils import write_summary_prompts, write_simulation_prompts
from utils.openai_api import summary_concept, simulate_cases


def concept_sum(all_contents, all_slot_attn, tokenizer, config, topk=10):
    # concept summary ____________________________________________
    config['llm_type'] = 'gpt-4o'
    write_summary_prompts(range(config['num_concepts']), np.array(all_contents), np.array(all_slot_attn), tokenizer,
                          config['prompt_path'], topk=topk, vis_threshold=config['vis_threshold'], max_len=config['max_len'])

    summary_list = []
    for concept_id in range(config['num_concepts']):
        s_prompt = open(config['prompt_path'] + 'summarization_concept_' + str(concept_id) + '.txt').read()
        summary = summary_concept(s_prompt, config['task'], num_response=1, max_tokens=100, model_type=config['llm_type'])
        summary_list.append(json.loads(summary.strip('```json\n')))
        print('concept ' + str(concept_id) + ': ' + summary)

    with open(config['prompt_path'] + "summary.json", 'w') as f:
        json.dump(summary_list, f, indent=4)


def concept_sim(all_contents, all_slot_attn, tokenizer, config, topk=100):
    summary_list = []
    with open(config['prompt_path'] + "summary.json", 'r') as f:
        summary = json.load(f)
    for item in summary:
        summary_list.append(item['Summary'])

    # concept simulation ____________________________________________
    config['llm_type'] = 'gpt-4o-mini'
    extraction_path = config['prompt_path'] + '/extraction/'
    simulation_path = config['prompt_path'] + '/simulation/'

    write_simulation_prompts(range(config['num_concepts']), np.array(all_contents), np.array(all_slot_attn), tokenizer,
                             extraction_path, summary_list, topk=topk, max_len=config['max_len'])

    simulate_cases(extraction_path, simulation_path, range(config['num_concepts']), summary_list, model_type=config['llm_type'])
