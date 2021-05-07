import torch
import sklearn.metrics as metric
THRESHOLD = 0.2

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        target = torch.argmax(target, dim=1)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        target = torch.topk(target, k, dim=1)[1]
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target[:, i]).item()
    return correct / len(target)


def precision(output, target):
    pred = torch.where(output > THRESHOLD, 1, 0)
    label = torch.where(target > 0.5, 1, 0)
    #improvised_nocall = torch.where(torch.sum(pred, dim=1) == 0, 1, 0)
    true_positives = torch.sum(pred * label)
    all_positives = torch.sum(pred)
    false_positives = all_positives - true_positives
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives/(true_positives+false_positives)
    # precision = metric.precision_score(pred.cpu(), label.cpu(), average='micro')
    return precision

def recall(output, target):
    pred = torch.where(output > THRESHOLD, 1, 0)
    label = torch.where(target > 0.5, 1, 0)
    true_positives = torch.sum(pred * label)
    z = torch.where(label - pred > 0, 1, 0)
    false_negatives = torch.sum(z)
    recall = true_positives/(true_positives+false_negatives)
    # recall = metric.recall_score(pred.cpu(), label.cpu(), average='micro')
    return recall

def f1(output, target):
    prec = precision(output, target)
    rec = recall(output, target)
    return 2*(prec*rec)/(prec+rec)


def tp(output, target):
    pred = torch.where(output > THRESHOLD, 1, 0)
    label = torch.where(target > 0.5, 1, 0)
    all_positives = torch.sum(pred)
    true_positives = torch.sum(pred * label)
    false_positives = all_positives - true_positives
    z = torch.where(label - pred > 0, 1, 0)
    false_negatives = torch.sum(z)
    true_negatives = (-1)*torch.sum((pred - 1))
    return true_positives

def fp(output, target):
    pred = torch.where(output > THRESHOLD, 1, 0)
    label = torch.where(target > 0.5, 1, 0)
    all_positives = torch.sum(pred)
    true_positives = torch.sum(pred * label)
    false_positives = all_positives - true_positives
    z = torch.where(label - pred > 0, 1, 0)
    false_negatives = torch.sum(z)
    true_negatives = (-1)*torch.sum((pred - 1))
    return false_positives

def fn(output, target):
    pred = torch.where(output > THRESHOLD, 1, 0)
    label = torch.where(target > 0.5, 1, 0)
    all_positives = torch.sum(pred)
    true_positives = torch.sum(pred * label)
    false_positives = all_positives - true_positives
    z = torch.where(label - pred > 0, 1, 0)
    false_negatives = torch.sum(z)
    true_negatives = (-1)*torch.sum((pred - 1))
    return false_negatives

def tn(output, target):
    pred = torch.where(output > THRESHOLD, 1, 0)
    label = torch.where(target > 0.5, 1, 0)
    all_positives = torch.sum(pred)
    true_positives = torch.sum(pred * label)
    false_positives = all_positives - true_positives
    z = torch.where(label - pred > 0, 1, 0)
    false_negatives = torch.sum(z)
    true_negatives = (-1)*torch.sum((pred - 1))
    return true_negatives
