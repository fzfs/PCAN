# -*- coding: utf-8 -*-

import util
import torch
import numpy as np
from model import Net
from tensorboardX import SummaryWriter

if __name__ == '__main__':    
    
    util.setup_seed(0)
    
    opt, fn = util.parser_model()
    lr = opt.lr
    avg = opt.avg
    alpha = opt.alpha
    weight = opt.weight
    epochs = opt.epochs
    network = opt.network
    pretrain = opt.pretrain
    truncated = opt.truncated
    init_type = opt.init_type
    image_size = opt.image_size
    batch_size = opt.batch_size
    class_name = opt.class_name
    milestones = opt.milestones
    weight_decay = opt.weight_decay

    writer = SummaryWriter(fn)
    gpu_ids = [g for g in range(torch.cuda.device_count())]

    train_loader, train_dataset_size = util.load_data('file/train.csv', 'train', batch_size, image_size, class_name)
    val_loader, val_dataset_size = util.load_data('file/val.csv', 'val', batch_size, image_size, class_name)

    model = Net(lr, weight_decay, init_type, gpu_ids, network, pretrain, avg, weight, milestones, truncated, alpha, len(class_name))
    
    model.print_networks()
    count = 0
    auc_temp = 0
    loss_temp = 100
    train_loss = 0.0
    tamp = train_dataset_size//3

    for epoch in range(epochs):
        model.train() 
        for i, train_data in enumerate(train_loader):
            model.set_input(train_data)
            model.optimize_parameters()
            train_loss += float(model.loss)

            if count % tamp == 0:
                with torch.no_grad():
                    val_loss = 0.0
                    val_loss1 = 0.0
                    val_loss2 = 0.0
                    model.eval()
                    for j, val_data in enumerate(val_loader):
                        model.set_input(val_data)
                        model('val')
                        val_loss += float(model.loss)
                        val_loss1 += float(model.loss1)
                        val_loss2 += float(model.loss2)
                        predicted, labels = model.predicted_val()
                        if j == 0:
                            p_all = predicted
                            l_all = labels
                        else:
                            p_all = np.vstack((p_all, predicted))
                            l_all = np.vstack((l_all, labels))
                    
                    mean = util.print_metrics1(l_all, p_all, mode='val', name=fn, class_name=class_name)
                        
                if count == 0:
                    util.cprint("Training: Epoch[{:0>2}/{:0>2}] Iteration[{:0>3}/{:0>3}] Train_Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, train_dataset_size, train_loss), fn + '/log.txt')
                
                else:
                    util.cprint("Training: Epoch[{:0>2}/{:0>2}] Iteration[{:0>3}/{:0>3}] Train_Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, train_dataset_size, train_loss/tamp), fn + '/log.txt')

                    writer.add_scalar('auc_weight', mean, count/tamp)
                    writer.add_scalars('train_val_loss', {'train_loss': train_loss / tamp,
                                                          'val_loss': val_loss / val_dataset_size}, count / tamp)
                    writer.add_scalars('val_loss', {'val_loss1': val_loss1 / val_dataset_size,
                                                          'val_loss2': val_loss2 / val_dataset_size}, count / tamp)

                model.train()
                train_loss = 0.0
        
                if mean > auc_temp:
                    auc_temp = mean
                    torch.save(model, fn+'/model_auc.pkl')
            
            count += 1
          
        model.update_learning_rate()
        writer.add_scalar('lr', float(model.lr), epoch)

    writer.close()
    torch.save(model, fn+'/model_final.pkl')
