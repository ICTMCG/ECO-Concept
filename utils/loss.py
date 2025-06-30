import torch


def distinctiveness_loss(data, att, topk=10):
    b1, c, d1 = data.shape
    record = []
    for i in range(c):
        current_f = data[:, i, :]
        current_att = att.sum(-1)[:, i]
        indices = sorted(range(b1), key=lambda i: current_att[i], reverse=True)[:topk]
        current_f = current_f[indices]
        record.append(torch.mean(current_f, dim=0, keepdim=True))
    record = torch.cat(record, dim=0)
    d_sim = torch.cdist(record[None, :, :], record[:, None, :])
    return d_sim.mean()


def consistency_loss(update, att, topk=10):
    b, cpt, spatial = att.size()
    consistence_loss = 0.0
    for i in range(cpt):
        current_up = update[:, i, :]
        current_att = att[:, i, :].sum(-1)
        indices = sorted(range(b), key=lambda i: current_att[i], reverse=True)[:topk]
        need = current_up[indices]
        d_sim = torch.cdist(need[None, :, :], need[:, None, :]).mean()
        consistence_loss += d_sim
    return consistence_loss / cpt
