import os
import re
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

client = OpenAI(
    base_url="",
    api_key='',
)

task_description = {
    'IMDB': 'whether the sentiment of movie reviews is positive or negative',
}

summary_instruction = open('./prompts/instruction_summarization.txt').read()
simulation_instruction = open('./prompts/instruction_simulation.txt').read()


def summary_concept(s_prompt, task, num_response=1, max_tokens=100, model_type='gpt-4o'):
    s_instruction = summary_instruction.replace('{{tasks}}', task_description[task])
    chat_messages = [{"role": "system", "content": s_instruction}, {"role": "user", "content": s_prompt}]
    ret = get_response(chat_messages, num_response, max_tokens, model_type)[0]
    return ret


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3), retry=(
        retry_if_exception_type(openai.RateLimitError) | retry_if_exception_type(openai.APIConnectionError)))
def get_response(chat_messages, num_response, max_tokens, model_type):
    _response = client.chat.completions.create(
        model=model_type,
        messages=chat_messages,
        temperature=0.1,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=num_response
    )
    all_responses = [_response.choices[i].message.content for i in
                     range(len(_response.choices))]
    return all_responses


def simulate_concept(cases_path, num_response=1, max_tokens=2000, model_type='gpt-4o-mini'):
    s_prompt = open(cases_path + '.txt').read()
    chat_messages = [{"role": "system", "content": simulation_instruction}, {"role": "user", "content": s_prompt}]
    ret = get_response(chat_messages, num_response, max_tokens, model_type)[0]
    return ret


def simulate_cases(extraction_path, simulation_path, concept_ids, concept_summaries, model_type='gpt-4o-mini'):
    if not os.path.exists(simulation_path):
        os.makedirs(simulation_path)

    for i in concept_ids:
        if concept_summaries[i] == 'No relation found':
            continue
        else:
            path = extraction_path + '/concept' + str(i)
            all_cases = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    match = re.search(r'case(\d+)', file)
                    if match:
                        all_cases.append(int(match.group(1)))

            for j in all_cases:
                ret = simulate_concept(path + '/case' + str(j), model_type=model_type)
                file_name = simulation_path + '/concept' + str(i)
                if not os.path.exists(file_name):
                    os.makedirs(file_name)
                with open(file_name + '/case' + str(j) + '.txt', 'w', encoding='utf-8') as file:
                    file.write(ret)



