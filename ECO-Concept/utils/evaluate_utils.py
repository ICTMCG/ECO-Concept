import os
import re
import pathlib
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import torch
import random
from tqdm import tqdm
import numpy as np
from utils.dataloader import make_dataset, TensorDataset
from torch.utils.data import DataLoader


def divide_cases(num_concepts, simulation_path, ratio=0.1):
    train_cases = dict.fromkeys(range(num_concepts), "default_value")
    val_cases = dict.fromkeys(range(num_concepts), "default_value")
    for concept_id in range(num_concepts):
        case = []
        for root, dirs, files in os.walk(simulation_path + '/concept' + str(concept_id)):
            for file in files:
                match = re.search(r'case(\d+)', file)
                if match:
                    case.append(int(match.group(1)))
        val_case = random.sample(case, int(len(case) * ratio))
        val_cases[concept_id] = val_case
        train_cases[concept_id] = [x for x in case if x not in val_case]
    return train_cases, val_cases


def get_simulation_scores(concept_id, case_id, clean_tokens, attention, simulation_path, max_len=256):
    simulation = open(simulation_path + '/concept' + str(concept_id) + '/case' + str(case_id) + '.txt').read()
    lines_s = simulation.split('\n')
    scores_s = [float(0)]
    for i in range(min(len(clean_tokens), len(lines_s), max_len - 1)):
        token_s, score_s = lines_s[i].strip().split()
        if clean_tokens[i].strip().lower() == token_s.strip().lower():
            scores_s.append(float(score_s))
        else:
            scores_s.append(attention[i + 1].item())
    scores_s = scores_s + [float(0)] * (max_len - len(scores_s))
    return scores_s


def evaluate(dl, model_new=None):
    loader = tqdm(dl, total=len(dl), unit="batches")
    model_new.eval()
    with torch.no_grad():
        loss_list = []
        batch_cnt = 0
        total_len = 0
        y_pred = []
        y_true = []
        for batch in loader:
            x, y, _ = batch
            prediction, loss, _, _, _ = model_new(x, y)
            batch_size, num_classes = prediction.shape
            y_pred.extend(prediction.cpu().numpy())
            y_list = y.cpu().numpy()
            for label in y_list:
                temp = [0 for _ in range(num_classes)]
                temp = [1 if i == label else 0 for i in range(len(temp))]
                y_true.append(temp)
            loss_list.append(loss)
            total_len += len(y)
            batch_cnt += 1

        all_metrics = metrics(y_true, y_pred, num_classes)
        all_loss = loss_format(loss_list)

    return all_loss, all_metrics


def metrics(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_metrics = {}
    all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)

    max_values = np.max(y_pred, axis=1, keepdims=True)
    y_pred = (y_pred == max_values).astype(int)
    y_pred = np.around(y_pred).astype(int)

    if num_classes == 2:
        all_metrics['f1'] = f1_score(y_true, y_pred, average='macro')
    else:
        predict_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        macro_f1_scores = []
        for label in range(num_classes):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
        all_metrics['f1'] = np.mean(macro_f1_scores)

    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    all_metrics['acc'] = accuracy_score(y_true, y_pred)
    if num_classes == 2:
        all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)
        all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)
        all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)
    return all_metrics


def loss_format(loss_list):
    mean_tensors = []
    for i in range(len(loss_list[0])):
        tensors_at_position = [tup[i] for tup in loss_list if i < len(tup)]
        mean_tensor = torch.mean(torch.stack(tensors_at_position), dim=0)
        mean_tensors.append(mean_tensor)

    all_loss = {}
    all_loss['Total loss'] = mean_tensors[0].item()
    all_loss['Pre loss'] = mean_tensors[1].item()
    all_loss['Distinctiveness loss'] = mean_tensors[2].item()
    all_loss['Consistence loss'] = mean_tensors[3].item()
    return all_loss


def all_evaluate(model, task, num_classes, batch_size, eval_type, bert_path, base_model_name, max_len=256):
    all_contents = []
    all_slot_attn = []

    data = make_dataset(pathlib.Path('./datasets/' + task + '/' + eval_type + '.json'))
    dataset = TensorDataset(data[0], data[1], bert_path, base_model_name=base_model_name, max_len=max_len)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loader = tqdm(dl, total=len(dl), unit="batches")
    with torch.no_grad():
        loss_list = []
        y_pred = []
        y_true = []
        for batch in loader:
            x, y, content = batch
            prediction, loss, slot_attn, concept_logits, _ = model(x, y)
            all_contents.extend(np.array(content))
            all_slot_attn.extend(slot_attn.cpu().detach().numpy())

            y_pred.extend(prediction.cpu().numpy())
            y_list = y.cpu().numpy()
            for label in y_list:
                temp = [0 for _ in range(num_classes)]
                temp = [1 if i == label else 0 for i in range(len(temp))]
                y_true.append(temp)

            loss_list.append(loss)

        all_loss = loss_format(loss_list)
        all_metrics = metrics(y_true, y_pred, num_classes)

    print(" ".join(("SCORES", "Loss", str(all_loss), "Metrics", str(all_metrics), "\n")))

    return all_contents, all_slot_attn, y_true, y_pred


def contains_replacement_character(text):
    for char in text:
        if ord(char) == 0xFFFD:
            return True
    return False

