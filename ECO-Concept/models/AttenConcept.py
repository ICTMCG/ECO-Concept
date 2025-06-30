import os
import torch
from tqdm.notebook import tqdm
from transformers import BertModel, RobertaModel, RobertaTokenizer, BertTokenizer
from models.slots import ConceptQuerySlotAttention, gru_cell
from utils.loss import distinctiveness_loss, consistency_loss
from utils.evaluate_utils import evaluate, loss_format, metrics, all_evaluate, divide_cases, get_simulation_scores


class AttenConcept(torch.nn.Module):
    def __init__(self, num_classes=2, num_concepts=20, dist_weight=0, con_weight=0, base_model_name='bert', bert_path=None,
                 dropout_prob=0.1, max_len=256):
        super().__init__()
        if base_model_name == 'bert':
            self.bert_model = BertModel.from_pretrained(bert_path)
        if base_model_name == 'roberta':
            self.bert_model = RobertaModel.from_pretrained(bert_path)
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.dist_weight = dist_weight
        self.con_weight = con_weight
        self.max_len = max_len

        self.num_heads = 8
        self.attention_dropout = 0.1
        self.projection_dropout = 0.1

        self.concept_slots_init = torch.nn.Embedding(self.num_concepts, self.embedding_dim)
        torch.nn.init.xavier_uniform_(self.concept_slots_init.weight)

        self.concept_slot_attention = ConceptQuerySlotAttention(slot_size=self.embedding_dim)

        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(self.num_concepts, self.num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.set_encoder_status(True)
        self.set_classifier_status(True)

    def set_encoder_status(self, status=True):
        num_encoder_layers = len(self.bert_model.encoder.layer)
        for (i, x) in enumerate(self.bert_model.encoder.layer):
            requires_grad = False
            if i == num_encoder_layers - 1:
                requires_grad = status
            for y in x.parameters():
                y.requires_grad = requires_grad

    def set_classifier_status(self, status=True):
        self.classifier.requires_grad = status

    def forward(self, x, y):
        content_feature = self.bert_model(x["input_ids"].squeeze(1).cuda(),
                                          x["attention_mask"].squeeze(1).cuda()).last_hidden_state

        concept_slots_init = self.concept_slots_init.weight.expand(content_feature.size(0), -1, -1)
        concepts, slot_attn = self.concept_slot_attention(content_feature, concept_slots_init,
                                                          x["attention_mask"].squeeze(1))

        concept_logits = torch.sum(slot_attn, dim=-1)
        output = self.classifier(concept_logits)
        prediction = torch.softmax(output.squeeze(1), dim=-1)
        pre_loss = self.loss_fn(output, y.cuda())

        # other losses
        d_loss = distinctiveness_loss(concepts, slot_attn)
        c_loss = consistency_loss(concepts, slot_attn)

        total_loss = pre_loss + self.dist_weight * d_loss + self.con_weight * c_loss

        return prediction, (total_loss, pre_loss, d_loss, c_loss), slot_attn, concept_logits, concepts


class Trainer:
    def __init__(self, config):
        self.config = config
        self.save_path = os.path.join(self.config['model_path'],
                                      self.config['task'] + ".pth")
        self.model = AttenConcept(num_classes=self.config['num_classes'],
                                  num_concepts=self.config['num_concepts'],
                                  dist_weight=self.config['dist_weight'],
                                  con_weight=self.config['con_weight'],
                                  base_model_name=self.config['base_model_name'],
                                  bert_path=self.config['bert_path'],
                                  max_len=self.config['max_len'])
        if self.config['base_model_name'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_path'])
        if self.config['base_model_name'] == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(self.config['bert_path'])

        self.mse_loss = torch.nn.MSELoss()
        if self.config['if_retrain']:
            self.model.load_state_dict(torch.load(config['checkpoint_path']))

    def freeze_embeddings(self, embedding_layer, c_ids):
        with torch.no_grad():
            for c_id in c_ids:
                embedding_layer.weight.grad[c_id] = 0

    def train(self, train_dl, val_dl, test_dl):
        torch.cuda.empty_cache()
        self.model.cuda()
        torch.cuda.empty_cache()

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])

        best_f1, best_epoch = 0, 0
        for epoch in range(self.config['epoch']):
            self.model.train()
            self.model.set_encoder_status(status=True)
            self.model.set_classifier_status(status=True)
            train_loader = tqdm(train_dl, total=len(train_dl), unit="batches", desc="training")
            for batch in train_loader:
                x, y, _ = batch
                prediction, loss, slot_attn, concept_logits, _ = self.model(x, y)
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()

            train_loss, train_metrics = evaluate(train_dl, self.model)
            print(" ".join(
                ("TRAIN SCORES" + " epoch", str(epoch), "Loss", str(train_loss), "Metrics", str(train_metrics), "\n")))

            val_loss, val_metrics = evaluate(val_dl, self.model)
            print(" ".join(
                ("VAL SCORES" + " epoch", str(epoch), "Loss", str(val_loss), "Metrics", str(val_metrics), "\n")))

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                test_loss, test_metrics = evaluate(test_dl, self.model)
                print(" ".join(
                    ("TEST SCORES" + " epoch", str(epoch), "Loss", str(test_loss), "Metrics", str(test_metrics), "\n")))
                torch.save(self.model.state_dict(), self.save_path)

        print("Best val f1: " + str(best_f1) + ", Best epoch: " + str(best_epoch))
        return test_metrics

    def retrain(self, train_dl, val_dl, test_dl, freeze_concepts=[]):
        train_cases, val_cases = divide_cases(self.config['num_concepts'], self.config['simulation_path'])

        self.model.cuda()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['retrain_lr'],
                                     weight_decay=self.config['weight_decay'])

        for epoch in range(self.config['epoch']):
            loss_list = []
            y_pred, y_true = [], []
            batch_cnt = 0

            self.model.train()
            self.model.set_encoder_status(status=False)
            self.model.set_classifier_status(status=True)
            self.model.concept_slots_init.requires_grad_(True)

            train_loader = tqdm(train_dl, total=len(train_dl), unit="batches", desc="training")

            for batch in train_loader:
                x, y, content = batch
                prediction, loss, slot_attn, concept_logits, features = self.model(x, y)
                batch_size, _, _ = slot_attn.shape

                batch_concept_loss = {}
                batch_concept_loss_val = {}
                for concept_id in range(self.config['num_concepts']):
                    batch_concept_loss[concept_id] = []
                    batch_concept_loss_val[concept_id] = []
                    for case_id in train_cases[concept_id] + val_cases[concept_id]:
                        if case_id in range(batch_cnt, batch_cnt + batch_size):
                            attention = slot_attn[case_id - batch_cnt][concept_id]
                            tokens = self.tokenizer.tokenize(content[case_id - batch_cnt])
                            clean_tokens = [self.tokenizer.convert_tokens_to_string(token) for token in tokens]
                            scores_s = get_simulation_scores(concept_id, case_id, clean_tokens, attention, self.config['simulation_path'])

                            if case_id in train_cases[concept_id]:
                                batch_concept_loss[concept_id].append(
                                    self.mse_loss(attention, torch.tensor(scores_s).cuda()))
                            elif case_id in val_cases[concept_id]:
                                batch_concept_loss_val[concept_id].append(
                                    self.mse_loss(attention, torch.tensor(scores_s).cuda()))

                contributions = []
                weights = list(self.model.classifier.parameters())[0]
                for i in range(batch_size):
                    contributions.append(concept_logits[i] * weights)
                contributions = torch.mean(torch.stack(contributions, dim=0), dim=0).T
                concept_contributions = []
                for concept_id in range(self.config['num_concepts']):
                    concept_contributions.append(round(sum([abs(x.item()) for x in contributions[concept_id]]), 3))

                i_loss = []
                i_loss_val = []
                for concept_id in range(self.config['num_concepts']):
                    if len(batch_concept_loss[concept_id]) != 0:
                        i_loss.append(concept_contributions[concept_id] * torch.mean(
                            torch.stack(batch_concept_loss[concept_id], dim=0), dim=0))
                    if len(batch_concept_loss_val[concept_id]) != 0:
                        i_loss_val.append(concept_contributions[concept_id] * torch.mean(
                            torch.stack(batch_concept_loss_val[concept_id], dim=0), dim=0))

                t_loss = loss[0]
                if len(i_loss) != 0:
                    t_loss += self.config['com_weight'] * torch.stack(i_loss).mean()
                    loss_list.append(loss + (torch.stack(i_loss).mean(),))
                else:
                    loss_list.append(loss)

                optimizer.zero_grad()
                t_loss.backward()
                self.freeze_embeddings(self.model.concept_slots_init, freeze_concepts)
                optimizer.step()

                y_pred.extend(prediction.cpu().detach().numpy())
                y_list = y.cpu().numpy()
                for label in y_list:
                    temp = [0 for _ in range(self.config['num_classes'])]
                    temp = [1 if i == label else 0 for i in range(len(temp))]
                    y_true.append(temp)

                batch_cnt += batch_size
                del x, y, prediction, loss, slot_attn, concept_logits, features
                torch.cuda.empty_cache()

            all_loss = loss_format(loss_list)
            all_metrics = metrics(y_true, y_pred, self.config['num_classes'])
            print(" ".join(("TRAIN SCORES", "Loss", str(all_loss), "Metrics", str(all_metrics), "\n")))
            val_loss, val_metrics = evaluate(val_dl, self.model)
            print(" ".join(
                ("VAL SCORES" + " epoch", str(epoch), "Loss", str(val_loss), "Metrics", str(val_metrics), "\n")))

            torch.save(self.model.state_dict(), self.config['model_path'] + str(epoch) + '.pth')

        test_loss, test_metrics = evaluate(test_dl, self.model)
        return test_metrics