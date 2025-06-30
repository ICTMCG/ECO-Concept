import argparse
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from utils.dataloader import make_dataset, TensorDataset
from models.AttenConcept import Trainer as ConceptTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--base_model_name', default='roberta')
parser.add_argument('--task', default='IMDB')
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--num_concepts", type=int, default=20)
parser.add_argument('--dist_weight', type=float, default=-0.01)
parser.add_argument('--con_weight', type=float, default=0.1)
parser.add_argument('--com_weight', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--retrain_lr', type=float, default=0.00001)
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--vis_threshold', type=float, default=0.3)
parser.add_argument('--llm_type', type=str, default='gpt-4o')
parser.add_argument("--data_path", type=str, default='./datasets/')
parser.add_argument("--bert_path", type=str, default='./pre-trained-models/')
parser.add_argument("--model_path", type=str, default='./checkpoints/')
parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/')
parser.add_argument("--prompt_path", type=str, default='./prompts/')
parser.add_argument("--simulation_path", type=str, default='./prompts/')
parser.add_argument('--if_retrain', type=bool, default=False)

ARGS = parser.parse_args()
seed = ARGS.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

config = {
    'base_model_name': ARGS.base_model_name,
    'task': ARGS.task,
    'num_classes': ARGS.num_classes,
    'num_concepts': ARGS.num_concepts,
    'dist_weight': ARGS.dist_weight,
    'con_weight': ARGS.con_weight,
    'com_weight': ARGS.com_weight,
    'batch_size': ARGS.batch_size,
    'lr': ARGS.lr,
    'retrain_lr': ARGS.retrain_lr,
    'seed': ARGS.seed,
    'epoch': ARGS.epoch,
    'max_len': ARGS.max_len,
    'weight_decay': 5e-5,
    'vis_threshold': ARGS.vis_threshold,
    'llm_type': ARGS.llm_type,
    'data_path': ARGS.data_path,
    'model_path': ARGS.model_path,
    'bert_path': ARGS.bert_path,
    'checkpoint_path': ARGS.checkpoint_path,
    'prompt_path': ARGS.prompt_path,
    'simulation_path': ARGS.simulation_path,
    'if_retrain': ARGS.if_retrain
}

if __name__ == '__main__':
    # Prepare Dataset ________________________________________________________
    train = make_dataset(pathlib.Path(ARGS.data_path + "train.json"))
    val = make_dataset(pathlib.Path(ARGS.data_path + "val.json"))
    test = make_dataset(pathlib.Path(ARGS.data_path + "test.json"))

    train_dataset = TensorDataset(train[0], train[1], ARGS.bert_path, ARGS.base_model_name, max_len=config['max_len'])
    val_dataset = TensorDataset(val[0], val[1], ARGS.bert_path, ARGS.base_model_name, max_len=config['max_len'])
    test_dataset = TensorDataset(test[0], test[1], ARGS.bert_path, ARGS.base_model_name, max_len=config['max_len'])

    train_dl = DataLoader(train_dataset, batch_size=ARGS.batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=ARGS.batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=ARGS.batch_size, shuffle=False)

    # Train Model ________________________________________________________
    trainer = ConceptTrainer(config)
    trainer.train(train_dl, val_dl, test_dl)

    # Finetune Model ________________________________________________________
    if config['if_retrain']:
        train_dl = DataLoader(train_dataset, batch_size=ARGS.batch_size, shuffle=False)
        trainer.retrain(train_dl, val_dl, test_dl)

