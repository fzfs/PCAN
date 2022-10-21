# -*- coding: utf-8 -*-

import util
from network_pretrain import Densenet121


def define_network(init_type, gpu_ids, network, pretrain=True, avg=0, weight=1, truncated=0, num_classes=5):
    net = Densenet121(pretrain, avg, weight, truncated, num_classes)
    
    return util.init_net(net, pretrain, init_type, gpu_ids)

