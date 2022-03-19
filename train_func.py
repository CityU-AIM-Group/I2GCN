import os
import time
import random
import numpy as np 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

from collections import OrderedDict, Counter
import torch
from torch.autograd import Variable

from utils import print_cz, time_mark, expand_user, model_snapshot, adjust_learning_rate, AverageMeter, accuracy, compute_precision, compute_sensitivity, compute_specificity, returnImportance
from loss_func import loss_dependence_batch, refine_ranking_loss, drop_consistency_loss

def train(
    train_loader,
    valid_loader, 
    test_loader, 
    extractor,
    aggregator,  
    aggregator_pre,   
    criterion, 
    optimizer_extractor, 
    optimizer_aggregator,
    optimizer_aggregator_pre, 
    args,
    logfile=None, 
    save_path=None, 
    start_epoch=0, 
    pth_prefix='', 
    curve_period = 10
    ):

    # optimizer.params = model.parameters()
    valid_select_loss = 10000

    for epoch in range(start_epoch, args.epochs):
        # lr
        adjust_learning_rate(optimizer_extractor, args.lr*args.lr_factor, epoch, args.lr_step, args.lr_gamma)
        adjust_learning_rate(optimizer_aggregator, args.lr, epoch, args.lr_step, args.lr_gamma)
        adjust_learning_rate(optimizer_aggregator_pre, args.lr, epoch, args.lr_step, args.lr_gamma)
        print_cz(str='Epoch:\t{:d}\t lr_ext:{:e}\t lr_agg:{:e}'.format(
            epoch, 
            optimizer_extractor.param_groups[0]['lr'], 
            optimizer_aggregator.param_groups[0]['lr']
            ), f=logfile)
        train_top1, train_loss = train_a_epoch(
            train_loader=train_loader, 
            extractor=extractor, 
            aggregator=aggregator,
            aggregator_pre=aggregator_pre, 
            criterion=criterion, 
            optimizer_extractor=optimizer_extractor, 
            optimizer_aggregator=optimizer_aggregator, 
            optimizer_aggregator_pre=optimizer_aggregator_pre,
            args=args,
            epoch=epoch, 
            logfile=logfile)

        ##########################################
        print_cz(" ==> Valid ", f=logfile)
        valid_top1, valid_loss, (_, _), (_, _)= test(
            extractor=extractor, 
            aggregator=aggregator, 
            aggregator_pre=aggregator_pre,
            test_loader=valid_loader,  
            criterion=criterion, 
            args=args,
            epoch=epoch, 
            logfile=logfile,
            test_flag='Valid')
        print_cz(" ", f=logfile)

        ##########################################
        print_cz(" ==> Test ", f=logfile)
        test_top1, test_loss, (test_pred_list, test_label_list), (test_output_list, test_target_list) = test(
            extractor=extractor, 
            aggregator=aggregator, 
            aggregator_pre=aggregator_pre,
            test_loader=test_loader,  
            criterion=criterion, 
            args=args,
            epoch=epoch, 
            logfile=logfile)
        print_cz(" ", f=logfile)
        print_cz('---', f=logfile)
        test_f1_macro = 100*metrics.f1_score(y_true=test_label_list, y_pred=test_pred_list, average='macro')
        print_cz(str='   F1 {:.3f}%'.format(
            test_f1_macro,
            ), f=logfile)
        #
        test_outputs = np.concatenate(test_output_list, axis=0)
        test_targets = np.concatenate(test_target_list, axis=0)
        test_auc = 100.0*roc_auc_score(y_true=test_targets, y_score=test_outputs, multi_class='ovr')
        print_cz(str='   AUC {:.3f}%'.format(test_auc), f=logfile)
        test_auc_ovo = 100.0*roc_auc_score(y_true=test_targets, y_score=test_outputs, multi_class='ovo')
        print_cz(str='   AUC {:.3f}%/{:.3f}%'.format(test_auc, test_auc_ovo), f=logfile)
        
        #
        confusion_matrix = metrics.confusion_matrix(y_true=test_label_list, y_pred=test_pred_list)
        precision_list = compute_precision(confusion_matrix)
        sensitivity_list = compute_sensitivity(confusion_matrix)
        specificity_list = compute_specificity(confusion_matrix)
        #
        print_cz(str='   Prec {:.3f}%\t Sen {:.3f}%\t Spec {:.3f}%'.format(
            precision_list[0],
            sensitivity_list[0],
            specificity_list[0]
            ), f=logfile) #########

        print_cz(str=confusion_matrix, f=logfile)
        print_cz(str=metrics.classification_report(y_true=test_label_list, y_pred=test_pred_list, digits=5), f=logfile)

        ##########################################
        # save valid-select model
        if valid_loss < valid_select_loss and epoch > int(args.epochs*0.65):
            if save_path is not None: # save flag
                model_snapshot(extractor, new_file=(
                    pth_prefix+'extractor-validselect-{}-acc{:.3f}-f{:.3f}-auc{:.3f}.pth'.format(
                        epoch,
                        test_top1, 
                        test_f1_macro, 
                        test_auc,
                        time_mark())
                    ), old_file=pth_prefix + 'extractor-validselect-', save_dir=save_path , verbose=True)
                model_snapshot(aggregator, new_file=(
                    pth_prefix+'aggregator-validselect-{}-acc{:.3f}-f{:.3f}-auc{:.3f}.pth'.format(
                        epoch,
                        test_top1, 
                        test_f1_macro, 
                        test_auc, 
                        time_mark())
                    ), old_file=pth_prefix + 'aggregator-validselect-', save_dir=save_path , verbose=True)
                model_snapshot(aggregator_pre, new_file=(
                    pth_prefix+'aggregator_pre-validselect-{}-acc{:.3f}-f{:.3f}-auc{:.3f}.pth'.format(
                            epoch,
                            test_top1, 
                            test_f1_macro,  
                            test_auc,
                            time_mark())
                    ), old_file=pth_prefix + 'aggregator_pre-validselect-', save_dir=save_path , verbose=True)
                valid_select_loss = valid_loss
            print_cz('** valid-select model saved successfully*', f=logfile)
        #  
        print_cz('---'*10, f=logfile)

def train_a_epoch(
    train_loader, 
    extractor, 
    aggregator, 
    aggregator_pre, 
    criterion, 
    optimizer_extractor, 
    optimizer_aggregator,
    optimizer_aggregator_pre,
    args,
    epoch, 
    logfile=None):

    extractor.cuda()
    extractor.train()
    aggregator.cuda()
    aggregator.train()
    aggregator_pre.cuda()
    aggregator_pre.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    epoch_start_time = time.time()
    end = time.time()

    for idx, (data, label, _) in enumerate(train_loader):
        extractor.zero_grad()
        aggregator.zero_grad()
        aggregator_pre.zero_grad()
        # input for extractor
        data_shape = data.shape # B*K*H*W
        data = data.view(data_shape[0]*data_shape[1], 1, data_shape[2], data_shape[3]) # (B*K)*1*H*W
        input_batch = Variable(data).cuda()
        label_batch = Variable(label).long().cuda()

        # data time
        data_time.update(time.time() - end)
        
        # extractor
        _, features, _ = extractor(x=input_batch) # (B*K)*512
        # features reshape
        features = features.view(data_shape[0], data_shape[1], -1) # B*Z*512
        features = torch.transpose(features, 1, 2) # B*512*Z
        #
        output_pre = aggregator_pre(features)
        loss_pre = criterion(output_pre, label_batch) * args.ratio_pre
  
        if args.job_type in ['S', 's']:
            _, importance_std = returnImportance(
                feature=features.clone().detach().data, 
                weight_softmax=aggregator_pre.state_dict()['classifier.weight'].data, 
                class_idx=[i for i in range(3)]) # B*K
        else:
            _, importance_std = returnImportance(
            feature=features.clone().detach().data, 
            weight_softmax=aggregator_pre.state_dict()['module.classifier.weight'].data, 
            class_idx=[i for i in range(3)]) # B*K
        importance_std = torch.unsqueeze(
                importance_std,
                dim=1)
        output, emb1, (bag_kept, bag_drop) = aggregator(
            features, 
            importances_batch=importance_std) # B*C*K, B*1*K

        # dropout_consistency loss
        loss_dc = drop_consistency_loss(bag_kept, bag_drop, size_average=True) * args.ratio_dc
        
        # HSIC loss
        interval = int(emb1.shape[1]/2)
        loss_HSIC = loss_dependence_batch(emb1[:, :interval,:], emb1[:, interval:,:]) * args.ratio_HSIC
        
        # refine rank loss
        logit_list = [output_pre, output]
        preds = []
        for i in range(label_batch.shape[0]): 
            pred = [logit[i][label_batch[i]] for logit in logit_list]
            preds.append(pred)
        loss_rank = refine_ranking_loss(preds, margin=args.rank_m, size_average=True) * args.ratio_rank

        # CE loss 
        loss = criterion(output, label_batch)      
        losses.update(loss.data.item(), label_batch.size(0))
        # performance
        prec1 = accuracy(output.data, label_batch.data, topk=(1,))[0]  
        top1.update(prec1[0], label_batch.size(0))

        # update param
        optimizer_extractor.zero_grad()
        optimizer_aggregator.zero_grad()
        optimizer_aggregator_pre.zero_grad()
        (loss + loss_pre + loss_dc + loss_HSIC + loss_rank).backward()
        optimizer_extractor.step()
        optimizer_aggregator.step()
        optimizer_aggregator_pre.step()

        del input_batch, label_batch
        del output
        del loss
        
        if idx % 1000 == 0:
            print_cz(' \t Batch {}, Train Loss {:.3f}  Prec@1 {:.3f}'.\
                format(idx, losses.value, top1.value), f=logfile)

        # batch time updated
        batch_time.update(time.time() - end)
        end = time.time()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    print_cz(str=' * Train time {:.3f}\t  BatchT: {:.3f}\t DataT: {:.3f}\t D/B: {:.1f}%'.format(epoch_time, batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)# print top*.avg
    print_cz(str='   Loss {:.3f}'.format(
            losses.avg
            ), 
        f=logfile)# print top*.avg
    print_cz(str='   Prec@1 {:.3f}%'.format(top1.avg), f=logfile)# print top*.avg

    return top1.avg, losses.avg, 


def test(
    extractor, 
    aggregator, 
    aggregator_pre,
    test_loader, 
    criterion,
    args, 
    epoch=0,  
    logfile=None,
    test_flag='Test'):

    extractor.cuda() 
    extractor.eval() 
    aggregator.cuda()
    aggregator.eval()
    aggregator_pre.cuda()
    aggregator_pre.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    label_list = []
    pred_list = []
    output_list = []
    target_list = []

    end = time.time()
    test_start_time = time.time()

    with torch.no_grad():
        for idx, (data, label, _) in enumerate(test_loader):
            # input for extractor
            data_shape = data.shape
            data = data.view(data_shape[0]*data_shape[1], 1, data_shape[2], data_shape[3])

            input_batch = Variable(data).cuda()
            label_batch = torch.autograd.Variable(label).long().cuda()
            # data time
            data_time.update(time.time() - end)
            # compute output
            _, features, _ = extractor(x=input_batch) # [B*K, C]
            features = features.view(data_shape[0], data_shape[1], -1) # [B, K, C]
            features = torch.transpose(features, 1, 2) # [B, C, K]
            #
            output_pre = aggregator_pre(features) # [B, 3], [B, C, K]
            loss_pre = criterion(output_pre, label_batch) * args.ratio_pre
       
            if args.job_type in ['S', 's']:
                _, importance_std = returnImportance(
                    feature=features.clone().detach().data, 
                    weight_softmax=aggregator_pre.state_dict()['classifier.weight'].data, 
                    class_idx=[i for i in range(3)]) # B*K
            else:
                _, importance_std = returnImportance(
                feature=features.clone().detach().data, 
                weight_softmax=aggregator_pre.state_dict()['module.classifier.weight'].data, 
                class_idx=[i for i in range(3)]) # B*K
            importance_std = torch.unsqueeze(
                importance_std, 
                dim=1) # [B, 1, K]
            output, emb1 = aggregator(
                features, 
                importances_batch=importance_std
            ) # B*C*K, B*1*K

            #
            output_list.append(torch.nn.functional.softmax(output, dim=-1).cpu().numpy())
            target_list.append(label.cpu().numpy())
                        
            # CE loss
            loss = criterion(output, label_batch)
            losses.update(loss.data.item(), label_batch.size(0))
            # performance
            prec1 = accuracy(output.data, label_batch.data, topk=(1,))[0]
            top1.update(prec1[0], label_batch.size(0))
            # update pred_list
            _, pred = output.topk(1, 1, True, True)#取maxk个预测值
            pred_list.extend(
                ((pred.cpu()).numpy()).tolist())
            # update label_list
            label_list.extend(
                ((label.cpu()).numpy()).tolist())
                                        
            del input_batch, label_batch
            del output
            del loss

            if idx % 100 == 0:
                print_cz(' \t Batch {}, Test Loss {:.3f}  Prec@1 {:.3f}'.\
                    format(idx, losses.value, top1.value), f=logfile)
            
            # batch time updated
            batch_time.update(time.time() - end)
            end = time.time()

    test_end_time = time.time()
    test_time = test_end_time - test_start_time

    print_cz(str=' * {}  time {:.3f}\t  BatchT: {:.3f}\t DataT: {:.3f}\t D/B: {:.1f}%'.format(test_flag, test_time, batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)# print top*.avg
    print_cz(str='   Loss {:.3f}'.format(
            losses.avg
            ), 
        f=logfile)# print top*.avg
    print_cz(str='   Prec@1 {:.3f}%'.format(top1.avg), f=logfile)# print top*.avg

    return top1.avg, losses.avg, (pred_list, label_list), (output_list, target_list)