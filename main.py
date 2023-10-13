import argparse
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
# from read_data import *
from read_data_cls import *
# from mixtext import MixText
import sys
sys.path.append('../')
from utils.util_training import set_parameter_learning_rate, load_model, save_model, setup_seed, to_tsne
from utils import architectures_label
import torch.optim as optim
import random


parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--arch', '-a', metavar='ARCH', default='bert')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=16, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lr', '--learning-rate-bert', default=5e-6, type=float,
                    metavar='LR', help='initial learning rate for bert')## yahoo:lr:1e-5,other:2MLP:5e-4
parser.add_argument('--other-lr', '--learning-rate-model', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=10,
                    help='number of labeled data')

parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')

parser.add_argument('--val-iteration', type=int, default=100,
                    help='number of labeled data')

parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix option, whether to mix or not')
parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix method, set different mix method')
parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='mix separate from labeled data and unlabeled data')
parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='set a random choice between mix and unmix during training')
parser.add_argument('--train-aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')

parser.add_argument('--model', type=str, default='../checkpoints/save_models/',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[0, 1, 2, 3], type=int, help='define mix layer set')

parser.add_argument('--alpha', default=0.1, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--lamda', default=0.5, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')
parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

parser.add_argument('--temp-change', default=1000000, type=int)
parser.add_argument('--margin', default=0.1, type=float, metavar='N',
                    help='margin for hinge loss')
parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='weight for hinge loss term of unlabeled data')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--training-seed', default=0, type=int)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument('--train-mode', default="hard", type=str, metavar='TYPE',
                        choices=['high', 'all', "hard", "easy", "margin","energy", "nouse","proto","randn"])
parser.add_argument('--ns-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'])
parser.add_argument('--rarm-up', default=False, type=bool, metavar='N',
                        help='warm-up weight')
parser.add_argument('--thread', default=0.85, type=float, metavar='WEIGHT',
                    help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
parser.add_argument('--use-labels', default=False, type=bool, metavar='N',
                    help='augment labeled training data')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("GPU num: ", n_gpu)

best_acc = 0
total_steps = 0
flag = 0
label_embed = torch.tensor([])
# print('Whether mix: ', args.mix_option)
# print("Mix layers sets: ", args.mix_layers_set)
mem_threshold = torch.tensor([])


def main():
    global label_embed
    global mem_threshold
    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, in_distribution_test_set, n_labels, in_distribution_n_labels,  test_set_tune, val_set_tune, label_emb = get_data_cls_tune1(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug, seed=args.seed)
    label_embed = label_emb
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True) ##labeled_data
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True) ##unlabeled_data
    # val_loader = Data.DataLoader(
    #     dataset=val_set, batch_size=512, shuffle=False)
    val_tune_loader = Data.DataLoader(
        dataset=val_set_tune, batch_size=512, shuffle=False)
    # test_loader = Data.DataLoader(
    #     dataset=test_set, batch_size=512, shuffle=False) ##all_acc
    in_distribution_test_loader = Data.DataLoader(
        dataset=in_distribution_test_set, batch_size=512, shuffle=True) ##in_acc
    test_tune_loader = Data.DataLoader(
        dataset=test_set_tune, batch_size=512, shuffle=True) ##ood_acc
    mem_threshold = torch.zeros(in_distribution_n_labels).cuda()


    def create_model():
        global label_embed
        print("=> creating model '{arch}'".format(
            arch=args.arch))
        model_factory = architectures_label.__dict__[args.arch]
        label_embed = label_embed.to(device)
        model_params = dict(pretrained=True, num_classes=in_distribution_n_labels, label_emb = label_embed, use_labels = args.use_labels)
        model = model_factory(**model_params)
        if(n_gpu>1):
            model = model.to(device)
            model = nn.DataParallel(model)
        else:
            model = model.to(device)
        return model
    # Load Model
    model = create_model()

    optimizer_grouped_parameters = set_parameter_learning_rate(model, args)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    test_accs = []
    in_distribution_test_accs = []
    out_distribution_test_accs = []
    out_distribution_test_accs_lof = []
    all_test_accs = []
    all_test_f1s = []
    all_roc_aucs = []
    all_test_accs_lof = []
    all_test_f1s_lof = []
    best_acc = 0.
    best_test_acc = [0.,0.,0.,0.,0.] #epoch,acc,f1,in_acc,ood_acc
    best_val_acc = [0.,0.,0.,0.,0.] #epoch,acc,f1,in_acc,ood_acc

    # # Start training
    for epoch in range(args.epochs):


        train(labeled_trainloader, unlabeled_trainloader, model, optimizer, epoch, in_distribution_n_labels, args.train_aug)
        test_to_tsne(val_tune_loader, model,epoch, threshold=0.5)
        '''
        val_loss, val_acc, _ = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')
        '''
        print ("val_result")
        val_acc, val_f1, out_acc, val_acc_lof, val_f1_lof = test_all_set(val_tune_loader, model)
        # if(epoch==10):
        #     test_to_tsne(val_tune_loader, model, threshold=0.5)
        print("epoch {}, val acc {}".format(epoch, val_acc))
        ################################################################################
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     best_val_acc = epoch, val_acc, val_f1, 0., out_acc
        # print('Best val acc(epoch,acc,f1,in_acc,ood_acc):')
        # print(best_val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            in_distribution_test_loss, in_distribution_test_acc, in_distribution_f1 = validate(
                in_distribution_test_loader, model)
            #out_test_acc, out_test_acc_lof = test_out_of_distribution_set(test_tune_loader, model)

            all_acc, all_f1, out_test_acc, all_acc_lof, all_f1_lof = test_all_set(test_tune_loader, model)
            # print ("=",all_roc_ood,out_test_acc)
            out_distribution_test_accs.append(out_test_acc)
            out_distribution_test_accs_lof.append(0.)
            all_test_accs.append(all_acc)
            all_test_f1s.append(all_f1)
            all_roc_aucs.append(0.)
            all_test_accs_lof.append(all_acc_lof)
            all_test_f1s_lof.append(all_f1_lof)
            test_accs.append(all_acc)
            in_distribution_test_accs.append(in_distribution_test_acc)
            best_test_acc = epoch, all_acc, all_f1, in_distribution_test_acc, out_test_acc
            # print ("save_model")
            # save_model(model,"best_seed"+str(args.training_seed)+"_labels"+str(in_distribution_n_labels)+".pkl")
        print('Best acc(epoch,acc,f1,in_acc,ood_acc):')
        print(best_test_acc)



    # print("Finished training!")
    # print('Best acc:')
    # print(best_test_acc)
    #
    # print('Test acc:')
    # print(test_accs)
    # #
    # print('In_distribution test acc:')
    # print(in_distribution_test_accs)



def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, epoch, n_labels,
          train_aug=False):
    # class_criterion = nn.BCELoss()
    unsup_criterion = nn.BCELoss()
    # ns_criterion = nn.BCELoss(reduction='none')
    if(args.ns_type == "mse"):
        ns_criterion = nn.MSELoss(reduction='none')
    else:
        ns_criterion = nn.BCELoss(reduction='none')
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    global total_steps
    global mem_threshold

    for batch_idx in range(args.val_iteration):

        total_steps += 1
        train_aug = args.train_aug

        if not train_aug:
            try:
                inputs_x, targets, inputs_x_length = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets, inputs_x_length = next(labeled_train_iter)
        else:
            try:
                (inputs_x, inputs_x_aug), (targets, _), (inputs_x_length,
                                                           inputs_x_length_aug) = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_aug), (targets, _), (inputs_x_length,
                                                           inputs_x_length_aug) = next(labeled_train_iter)



        # train_aug = True
        try:
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2, length_ori) = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2, length_ori) = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_ori.size(0)
        # targets_x = torch.zeros(batch_size, n_labels).scatter_(
        #     1, targets_x.view(-1, 1), 1)
        targets_x = torch.eye(n_labels)[targets, :]
        # import pdb
        # pdb.set_trace()
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        targets = targets.cuda()
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_ori = inputs_ori.cuda()


        outputs_x, token_emb, label_emb = model(inputs_x,mode="LT")

        outputs_u, u_emb = model(inputs_u)
        outputs_u2, u2_emb = model(inputs_u2)
        with torch.no_grad():
            outputs_ori, sup_emb = model(inputs_ori)
        sup_loss = torch.mean(torch.sum(-torch.log(outputs_x + 1e-8) * targets_x, 1)) \
                       + torch.mean(torch.max(-torch.log(1 - outputs_x + 1e-8) * (1 - targets_x), 1)[0]) ##运行yahooSSHI
        # sup_loss = class_criterion(outputs_x, targets_x)
        # unsup_loss = class_criterion(outputs_u, outputs_ori.clone().detach())
        # unsup_loss = 0.5 * class_criterion(outputs_u, outputs_ori.clone().detach()) + 0.5 * class_criterion(outputs_u2,\
        #                                                                                                     outputs_ori.clone().detach())
        # unsup_loss = 0.5 * unsup_criterion(outputs_u, outputs_ori.clone().detach()) + 0.5 * unsup_criterion(outputs_u2,\
        #                                                                                                     outputs_ori.clone().detach())
        # unsup_loss = 0.5 * class_criterion(outputs_u2, outputs_u.clone().detach()) + 0.5 * class_criterion(outputs_u, outputs_u2.clone().detach())
        # if(args.ns_type == "mse"):
        # unsup_loss = unsup_criterion(outputs_u, outputs_u2.detach()) + unsup_criterion(outputs_u2, outputs_u.detach())
        unsup_loss = unsup_criterion(outputs_u, outputs_ori.detach())
        # + unsup_criterion( 1- outputs_u, (1 - outputs_ori).detach())
        # else:
        #     unsup_loss = unsup_criterion(outputs_u, outputs_ori.detach())
            # unsup_loss = 0.5 * unsup_criterion(outputs_u2, outputs_u.clone().detach()) + 0.5 * unsup_criterion(
            #     outputs_u, outputs_u2.clone().detach())
        if (train_aug):
            inputs_x_aug = inputs_x_aug.cuda()
            outputs_x_a, _ = model(inputs_x_aug)
            unsup_loss += unsup_criterion(outputs_x_a, outputs_x)



        ns_target = torch.zeros_like(outputs_u)
        with torch.no_grad():
            # E_u = torch.mean(sup_emb, dim=0)
            #pseudo label
            prob, p_y = torch.max(outputs_u.data, 1)
            neg_emb = torch.cat([sup_emb, token_emb], 
                                dim=0)
            pseudo = torch.eye(n_labels)[p_y, :].cuda()
            prob_mask = (prob > 0.5).type(torch.int64).unsqueeze(1).expand(prob.size(0),n_labels)
            p_y = torch.where(prob > 0.5 ,prob , prob_mask)
            un_flag = 1 - pseudo * prob_mask

            neg_flag = torch.cat([un_flag, 1 - targets_x],
                                 dim=0)
            flag = neg_flag / sum(neg_flag)
            # print (un_flag,p_y)
            En = torch.mm(flag.T, neg_emb)
            # pdb.set_trace()
            ###positive
            flag_p = targets_x / (sum(targets_x) + 1e-8)
            Ep = torch.mm(flag_p.T, token_emb)

            # lamda = args.lamda
            # En = E_u.unsqueeze(0).expand_as(En_l) * lamda + En_l * ( 1 - lamda )
            cos_thread = torch.cosine_similarity(sup_emb.unsqueeze(1), En.unsqueeze(0), dim=2)

            cos_p = torch.cosine_similarity(sup_emb.unsqueeze(1), Ep.unsqueeze(0), dim=2)
            mask =  (cos_thread > cos_p).type(torch.int64)  * (cos_thread > args.thread).type(torch.int64)
            # mask = (cos_thread > args.thread).type(torch.int64)
            # print (mask,p_y)
            ###balacne each class

            random_index = torch.randint(1, 999, (mask.size(0), mask.size(1))).cuda()
            random_index = random_index * mask
            balance_index = torch.min(sum(mask))
            # # pdb.set_trace()
            kindex = torch.max(batch_size_2 - balance_index, torch.tensor(1).cuda())
            class_thread = torch.kthvalue(random_index, kindex, dim=0)[0]
            class_mask = random_index.gt(class_thread).type(torch.int64)

            k_mask = (sum(targets_x)>0).type(torch.int64)
            k_mask =  k_mask.unsqueeze(0).expand_as(mask)



        nl_loss = torch.sum(ns_criterion(outputs_u, ns_target) * k_mask * class_mask) / torch.max(
            torch.sum(k_mask * class_mask), torch.tensor(1.).cuda())

      
        weight = args.alpha
        loss = sup_loss + unsup_loss + args.alpha * nl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 1000 == 0:
            # print("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
            #     epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))
            print(
                "epoch:{:.4f}, step:{:.4f}, loss {:.4f},unsup_loss {:.4f},sup_loss {:.4f}, nl_loss: {:.4f},weight:{} ".format(
                    epoch,
                    total_steps,
                    loss.item(),
                    unsup_loss.item(),
                    sup_loss.item(),
                    nl_loss,
                weight))

def get_tsa_thresh(global_step, num_train_steps, start, end, schedule = "linear_schedule"):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress

    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    # print("linear_schedule", training_progress * (end - start) + start)
    # print("exp_schedule", torch.exp((training_progress - 1) * scale) * (end - start) + start)
    # print("log_schedule", (1 - torch.exp((-training_progress) * scale)) * (end - start) + start)
    return output

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def ramp_up(epoch, max_epochs, max_val, mult = 2):
    if epoch == 0:
        return 0.

    elif epoch >= max_epochs:
        return max_val
    # scale = 5
    # training_progress = float(epoch) / max_epochs
    # threshold = 1 - np.exp((-training_progress) * scale)
    # return threshold * max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)

def compute_loss(y_pred,lamda=0.05):
    row = torch.arange(0,y_pred.shape[0],3,device='cuda') # [0,3]
    col = torch.arange(y_pred.shape[0], device='cuda') # [0,1,2,3,4,5]
    col = torch.where(col % 3 != 0)[0].cuda() # [1,2,4,5]
    y_true = torch.arange(0,len(col),2,device='cuda')
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = torch.index_select(similarities,0,row)
    similarities = torch.index_select(similarities,1,col)
    similarities = similarities / lamda
    loss = F.cross_entropy(similarities,y_true)
    return torch.mean(loss)



def validate(valloader, model):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        pred_list = []
        target_list = []

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs, _ = model(inputs)
            # loss = criterion(outputs, targets)

            prob, predicted = torch.max(outputs.data, 1)

            # if batch_idx == 0:
            #     print("in_acc:Sample some true labeles and predicted labels")
            #     print(targets[:20])
            #     print(predicted[:20])
            #     print (prob[:20])

            pred_list.extend(np.array(predicted.cpu()))
            target_list.extend(np.array(targets.cpu()))
            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            # loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample
        f1 = f1_score(target_list, pred_list, average='macro')

    return loss_total, acc_total, f1



def test_out_of_distribution_set(eval_data, model, threshold = 0.5):
    model.eval()

    with torch.no_grad():
        correct = 0
        total_sample = 0
        clf = LocalOutlierFactor(n_neighbors=20, leaf_size=20)
        correct_lof = 0
        max_energy_correct = 0
        sum_energy_correct = 0
        for batch_idx, (inputs, targets, targets_tune, length) in enumerate(eval_data):
            # meters.update('data_time', time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets_tune = targets_tune.cuda()
            # compute output
            output, _ = model(inputs)


            probs, predicted = torch.max(output.data, 1)
            predicted[torch.where(probs.ge(threshold))] = 1
            predicted[torch.where(probs.lt(threshold))] = 0
            # predicted[torch.where(prob.ge(threshold))] = 1
            # predicted[torch.where(prob.lt(threshold))] = 0
            # if batch_idx == 0:
            #     print("ood_acc:Sample some true labeles and predicted labels")
            #     print(targets_tune[:20])
            #     print(predicted[:20])
            #     print(probs[:20])



            correct += (np.array(predicted.cpu()) == np.array(targets_tune.cpu())).sum()  # in the target_tune is 0 or 1
            # import pdb
            # pdb.set_trace()
            # targets[torch.where(targets.eq(0))] = -1  # change 0 to -1

            # correct_lof += (np.array(predicted_out_lof) ==
            #                 np.array(targets.cpu())).sum()
            total_sample += inputs.shape[0]
        acc_total = correct / total_sample
        acc_total_lof = correct_lof / total_sample


    return acc_total, acc_total_lof

def test_to_tsne(testloader, model, epoch, threshold=0.5):
    model.eval()
    with torch.no_grad():
        embed_list = []
        label_list = []
        for batch_idx, (inputs, targets, targets_tune, length) in enumerate(testloader):
            inputs, targets, targets_tune = inputs.cuda(), targets.cuda(non_blocking=True), targets_tune.cuda(non_blocking=True)
            outputs, embedding = model(inputs)
            embed_list.extend(embedding.cpu().detach().numpy().tolist())
            label_list.extend(np.array(targets.cpu()).tolist())
            # pred_tune_list.extend(np.array(predicted_out))
            # pred_score_list.extend(np.array(confidence_score))
        # acc_total = correct / total_sample
    # to_tsne(embed_list, label_list, 2, "./version1/ag_ours_"+str(epoch)+".pdf")

def test_all_set(testloader, model, threshold=0.5):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        pred_list = []
        target_list = []
        pred_tune_list = []
        target_tune_list = []
        for batch_idx, (inputs, targets, targets_tune, idx) in enumerate(testloader):
            inputs, targets, targets_tune = inputs.cuda(), targets.cuda(non_blocking=True), targets_tune.cuda(non_blocking=True)
            outputs, embedding = model(inputs)
            probs, predicted = torch.max(outputs.data, 1)
            # predicted_out_lof = clf.fit_predict(embedding.cpu())

            predicted[torch.where(probs.lt(threshold))] = -1
            predicted_tune = torch.clone(predicted)
            predicted_tune[torch.where(probs.lt(threshold))] = 0
            predicted_tune[torch.where(probs.ge(threshold))] = 1

            # if batch_idx == 0:
            #     print("all_acc:Sample some true labeles and predicted labels")
            #     print(targets[:20])
            #     print(predicted[:20])
            #     print (probs[:20])
            #     # print (idx)
            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            total_sample += inputs.shape[0]
            pred_list.extend(np.array(predicted.cpu()))
            # pred_lof_list.extend(np.array(predicted_lof.cpu()))
            target_list.extend(np.array(targets.cpu()))
            pred_tune_list.extend(np.array(predicted_tune.cpu()))
            target_tune_list.extend(np.array(targets_tune.cpu()))
            # pred_tune_list.extend(np.array(predicted_out))
            # pred_score_list.extend(np.array(confidence_score))
        # acc_total = correct / total_sample
        f1 = f1_score(target_list, pred_list, average='macro')
        # f1_lof = f1_score(target_list, pred_lof_list, average='macro')
        # recall = recall_score(target_list, pred_list, average='macro')
        # precis = precision_score(target_list, pred_list, average='macro')
        # try:
        #     roc_auc = roc_auc_score(pred_tune_list, pred_score_list, average='macro')
        # except ValueError:
        #     roc_auc = 0

        accuracy = accuracy_score(target_list, pred_list)
        accuracy_ood = accuracy_score(pred_tune_list, target_tune_list)

    return accuracy, f1, accuracy_ood, 0., 0.



if __name__ == '__main__':
    setup_seed(args.training_seed)
    main()
