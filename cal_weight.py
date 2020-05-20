# coding: utf-8

import numpy as np
import os
from keras import backend as K

def get_slope(output, slope):
    conf = []
    for sample in output:
        conf.append([slope if num < 0 else 1 for num in sample])
    conf = np.array(conf)
    return conf
	
def get_weight(model, data):
    weights = np.array(model.get_weights())
    get_bn1_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[2].output])
    # output in test mode = 0
    bn1_output = get_bn1_layer_output([data, 0])[0]

    get_bn2_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[6].output])
    bn2_output = get_bn2_layer_output([data, 0])[0]

    relu1_slope = get_slope(bn1_output, 0.3)
    relu2_slope = get_slope(bn2_output, 0.3)

    all_W = []
    for i in range(len(relu1_slope)):
        W1 = weights[0].T
        W2 = weights[6].T * relu1_slope[i]*weights[2]/np.sqrt(weights[5]+0.001)
        W3 = weights[12].T * relu2_slope[i]*weights[8]/np.sqrt(weights[11]+0.001)
        W_im = np.matmul(W3, W2)
        W = np.matmul(W_im, W1)
        all_W.extend(W)
    all_W = np.array(all_W)
    return all_W