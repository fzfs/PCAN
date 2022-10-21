# -*- coding: utf-8 -*-

import torch
import util
import numpy as np

if __name__ == '__main__':    
        
    opt, fn = util.parser_model('test')

    avg = opt.avg
    weight = opt.weight
    auc_loss = opt.auc_loss
    init_type = opt.init_type
    image_size = opt.image_size
    batch_size = opt.batch_size
    class_name = opt.class_name
    gpu_ids = [g for g in range(torch.cuda.device_count())]
    test_loader, test_dataset_size = util.load_data('file/test.csv', 'test', batch_size, image_size, class_name)
    
    if auc_loss == 'auc':
        model_path = fn+'/model_auc.pkl'
    else:
        model_path = fn+'/model_final.pkl'
        
    model = torch.load(model_path)
    
    print('Test')
    if avg == 0:
        with torch.no_grad():
            test_loss = 0.0
            model.eval()
            for k, test_data in enumerate(test_loader):
                model.set_input(test_data)
                model('test')
                test_loss += float(model.loss)
                predicted, predicted_all, labels, weights, im = model.predicted_test()
                if k == 0:
                    p_all = predicted
                    l_all = labels
                    
                else:
                    p_all = np.concatenate((p_all, predicted), axis=0)
                    l_all = np.concatenate((l_all, labels), axis=0)

            _ = util.print_metrics1(l_all, p_all, mode='test', name=fn, class_name=class_name, auc_loss=auc_loss)
            
    else:
        with torch.no_grad():
            test_loss = 0.0
            model.eval()
            for k, test_data in enumerate(test_loader):
                model.set_input(test_data)
                model('test')
                
                test_loss += float(model.loss)
                predicted, labels = model.predicted_val()
                if k == 0:
                    p_all = predicted
                    l_all = labels
                else:
                    p_all = np.concatenate((p_all, predicted), axis=0)
                    l_all = np.concatenate((l_all, labels), axis=0)  
            _ = util.print_metrics1(l_all, p_all, mode='test', name=fn, class_name=class_name, auc_loss=auc_loss)
