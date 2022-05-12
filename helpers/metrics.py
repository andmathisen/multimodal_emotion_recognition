import torch 
import numpy as np
import math
from sklearn.metrics import mean_absolute_error,mean_squared_error,precision_recall_fscore_support,accuracy_score

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    _, preds = torch.max(output, 1)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def zero_rule_algorithm_classification(train, val):
    
    output_values = [row[-1] for row in train]
    output_values = torch.cat(output_values).tolist()
    
    prediction_nr = max(set(output_values), key=output_values.count)
    targets = [row[-1] for row in val]
    targets = torch.cat(targets).tolist()
    preds = torch.full((len(targets),), prediction_nr)
    
    
    mse = math.sqrt(mean_squared_error(targets,preds))
    mae = mean_absolute_error(targets,preds)

    return mse,mae

def metrics_from_cm(confusion_matrices):
    precisions = []
    recalls = []
    f1_scores = []
    for matrix in confusion_matrices:
        matrix = np.asarray(matrix)
        tp_fn = matrix.sum(axis=1)
        tp_fp = matrix.sum(axis=0)
        tp = matrix.diagonal()
        precision = tp/tp_fp
        precision = np.nan_to_num(precision)
        recall = tp/tp_fn
        recall = np.nan_to_num(recall)

        f1 = 2*((precision*recall)/(precision+recall))
        f1 = np.nan_to_num(f1)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return np.mean(precisions,axis=0),np.mean(recalls,axis=0),np.mean(f1_scores,axis=0)

def mcc(c,s,p_k_list,t_k_list): # Matthews Correlation Coefficient multi-class case
    return(((c*s)-(sum([a * b for a, b in zip(p_k_list, t_k_list)])))/(math.sqrt(((s**2)-(sum([x**2 for x in p_k_list])))*((s**2)-(sum([x**2 for x in t_k_list]))))))


def calculate_metrics(output, target):
    
    output = torch.argmax(output,dim=1)
    output = output.tolist()
    target = target.tolist()
    acc = accuracy_score(output,target)
    prec,recall,f1_score,support = precision_recall_fscore_support(output,target,labels = [0,1,2,3,4,5,6],zero_division=0)
    return acc,prec,recall,f1_score,support
    
class Loss(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sumLoss = 0
        self.avgLoss = 0
        self.count = 0
    def update(self, loss, n=1):
        
        self.sumLoss += loss * n
        self.count += n
        self.avgLoss = self.sumLoss / self.count


class Metrics(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        
        
        self.avgAcc = 0
        self.avgPrec = [0,0,0,0,0,0,0]
        self.avgRecall = [0,0,0,0,0,0,0]
        self.avgF1 = [0,0,0,0,0,0,0]
        self.sumAcc = 0
        self.sumPrec = [0,0,0,0,0,0,0]
        self.sumRecall = [0,0,0,0,0,0,0]
        self.sumF1 = [0,0,0,0,0,0,0]
        self.count = 0
        self.totSupport = [0,0,0,0,0,0,0]


    def update(self, acc,prec,recall,f1_score,support, n=1):
        
        self.sumAcc += acc * n
        
        self.totSupport = [x+y for x,y in zip(self.totSupport,support)]
        
        tmpPrec = [x*y for x,y in zip(prec,support)]
        
        self.sumPrec = [x+y for x,y in zip(self.sumPrec,tmpPrec)]
        tmpRecall = [x*y for x,y in zip(recall,support)]
        self.sumRecall = [x+y for x,y in zip(self.sumRecall,tmpRecall)]
        tmpF1 = [x*y for x,y in zip(f1_score,support)]
        self.sumF1 = [x+y for x,y in zip(self.sumF1,tmpF1)]
        self.count += n
        self.avgAcc = self.sumAcc / self.count
        self.avgPrec = [np.nan_to_num(np.divide(x,int(y))) for x,y in zip(self.sumPrec,self.totSupport)]
        self.avgRecall = [np.nan_to_num(np.divide(x,int(y))) for x,y in zip(self.sumRecall,self.totSupport)]
        self.avgF1 = [np.nan_to_num(np.divide(x,int(y))) for x,y in zip(self.sumF1,self.totSupport)]

