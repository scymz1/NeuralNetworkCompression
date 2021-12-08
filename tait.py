#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import argparse
import torch
import numpy as np
import torch.nn as nn
import math
from utils import *
from mobilenetv2 import *
from quantize import *

# model settings
#torch.cuda.set_device(0)
torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser(description="MobileNetv2 Tunable Activation Imbalance Transfer Quantization")
parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10', 'fire', 'dog_cat'],
                        help='type of dataset')

parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')

parser.add_argument("--nw", type=int, default=8)
parser.add_argument("--na", type=int, default=8)
parser.add_argument("--nb", type=int, default=24)
parser.add_argument("--nm", type=int, default=24)
parser.add_argument("--t", type=float, default=0.2)


def inference(x, conv_layers, QReLUs):

    # init_block
    x = conv_layers[0](x)
    x = QReLU(x, QReLUs[0])

    # stage1 unit1
    x = conv_layers[1](x)
    x = QReLU(x, QReLUs[1])
    x = conv_layers[2](x)
    x = QReLU(x, QReLUs[2])
    x = conv_layers[3](x)

    # stage2 unit1
    x = conv_layers[4](x)
    x = QReLU(x, QReLUs[4])
    x = conv_layers[5](x)
    x = QReLU(x, QReLUs[5])
    x = conv_layers[6](x)

    # stage2 unit2
    x_ = x
    x = conv_layers[7](x_)
    x = QReLU(x, QReLUs[7])
    x = conv_layers[8](x)
    x = QReLU(x, QReLUs[8])
    x = conv_layers[9](x)
    x = x + x_

    # stage3 unit1
    x = conv_layers[10](x)
    x = QReLU(x, QReLUs[10])
    x = conv_layers[11](x)
    x = QReLU(x, QReLUs[11])
    x = conv_layers[12](x)
    
    # stage3 unit2
    x_ = x
    x = conv_layers[13](x_)
    x = QReLU(x, QReLUs[13])
    x = conv_layers[14](x)
    x = QReLU(x, QReLUs[14])
    x = conv_layers[15](x)
    x_ = x + x_

    # stage3 unit3
    x = conv_layers[16](x_)
    x = QReLU(x, QReLUs[16])
    x = conv_layers[17](x)
    x = QReLU(x, QReLUs[17])
    x = conv_layers[18](x)
    x = x + x_

    # stage4 unit1
    x = conv_layers[19](x)
    x = QReLU(x, QReLUs[19])
    x = conv_layers[20](x)
    x = QReLU(x, QReLUs[20])
    x = conv_layers[21](x)
    x_ = x

    # stage4 unit2
    x = conv_layers[22](x_)
    x = QReLU(x, QReLUs[22])
    x = conv_layers[23](x)
    x = QReLU(x, QReLUs[23])
    x = conv_layers[24](x)
    x_ = x + x_
    
    # stage4 unit3
    x = conv_layers[25](x_)
    x = QReLU(x, QReLUs[25])
    x = conv_layers[26](x)
    x = QReLU(x, QReLUs[26])
    x = conv_layers[27](x)
    x_ = x + x_

    # stage4 unit4
    x = conv_layers[28](x_)
    x = QReLU(x, QReLUs[28])
    x = conv_layers[29](x)
    x = QReLU(x, QReLUs[29])
    x = conv_layers[30](x)
    x = x + x_

    # stage4 unit5
    x = conv_layers[31](x)
    x = QReLU(x, QReLUs[31])
    x = conv_layers[32](x)
    x = QReLU(x, QReLUs[32])
    x = conv_layers[33](x)

    # stage4 unit6
    x_ = x
    x = conv_layers[34](x_)
    x = QReLU(x, QReLUs[34])
    x = conv_layers[35](x)
    x = QReLU(x, QReLUs[35])
    x = conv_layers[36](x)
    x_ = x + x_

    # stage4 unit7
    x = conv_layers[37](x_)
    x = QReLU(x, QReLUs[37])
    x = conv_layers[38](x)
    x = QReLU(x, QReLUs[38])
    x = conv_layers[39](x)
    x = x + x_

    # stage5 unit1
    x = conv_layers[40](x)
    x = QReLU(x, QReLUs[40])
    x = conv_layers[41](x)
    x = QReLU(x, QReLUs[41])
    x = conv_layers[42](x)
    
    # stage5 unit2
    x_ = x
    x = conv_layers[43](x_)
    x = QReLU(x, QReLUs[43])
    x = conv_layers[44](x)
    x = QReLU(x, QReLUs[44])
    x = conv_layers[45](x)
    x_ = x + x_

    # stage5 unit3
    x = conv_layers[46](x_)
    x = QReLU(x, QReLUs[46])
    x = conv_layers[47](x)
    x = QReLU(x, QReLUs[47])
    x = conv_layers[48](x)
    x = x + x_

    # stage5 unit4
    x = conv_layers[49](x)
    x = QReLU(x, QReLUs[49])
    x = conv_layers[50](x)
    x = QReLU(x, QReLUs[50])
    x = conv_layers[51](x)

    # final_block
    x = conv_layers[52](x)
    x = QReLU(x, QReLUs[52])
    # final_pool
    x = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)(x)
    # output
    x = conv_layers[53](x)
    x = x.view(x.size(0), -1)

    return x

def qinference(x, conv_layers, SW, SA):
    # init_block
    x = qconv(x, conv_layers[0], SW[0], SA[0])
    x = nn.ReLU6(inplace=True)(x)
    
    # stage1 unit1
    x = qconv(x, conv_layers[1], SW[1], SA[1])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[2], SW[2], SA[2])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[3], SW[3], SA[3])

    # stage2 unit1
    x = qconv(x, conv_layers[4], SW[4], SA[4])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[5], SW[5], SA[5])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[6], SW[6], SA[6])

    # stage2 unit2
    x_ = x
    x = qconv(x_, conv_layers[7], SW[7], SA[7])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[8], SW[8], SA[8])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[9], SW[9], SA[9])
    x = x + x_

    # stage3 unit1
    x = qconv(x, conv_layers[10], SW[10], SA[10])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[11], SW[11], SA[11])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[12], SW[12], SA[12])
    
    # stage3 unit2
    x_ = x
    x = qconv(x_, conv_layers[13], SW[13], SA[13])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[14], SW[14], SA[14])
    x = nn.ReLU6(inplace=True)(x) 
    x = qconv(x, conv_layers[15], SW[15], SA[15])
    x_ = x + x_

    # stage3 unit3
    x = qconv(x_, conv_layers[16], SW[16], SA[16])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[17], SW[17], SA[17])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[18], SW[18], SA[18])
    x = x + x_

    # stage4 unit1
    x = qconv(x, conv_layers[19], SW[19], SA[19])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[20], SW[20], SA[20])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[21], SW[21], SA[21])
    x_ = x

    # stage4 unit2
    x = qconv(x_, conv_layers[22], SW[22], SA[22])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[23], SW[23], SA[23])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[24], SW[24], SA[24])
    x_ = x + x_
    
    # stage4 unit3
    x = qconv(x_, conv_layers[25], SW[25], SA[25])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[26], SW[26], SA[26])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[27], SW[27], SA[27])
    x_ = x + x_

    # stage4 unit4
    x = qconv(x_, conv_layers[28], SW[28], SA[28])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[29], SW[29], SA[29])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[30], SW[30], SA[30])
    x = x + x_

    # stage4 unit5
    x = qconv(x, conv_layers[31], SW[31], SA[31])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[32], SW[32], SA[32])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[33], SW[33], SA[33])

    # stage4 unit6
    x_ = x
    x = qconv(x_, conv_layers[34], SW[34], SA[34])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[35], SW[35], SA[35])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[36], SW[36], SA[36])
    x_ = x + x_

    # stage4 unit7
    x = qconv(x_, conv_layers[37], SW[37], SA[37])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[38], SW[38], SA[38])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[39], SW[39], SA[39])
    x = x + x_

    # stage5 unit1
    x = qconv(x, conv_layers[40], SW[40], SA[40])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[41], SW[41], SA[41])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[42], SW[42], SA[42])
    
    # stage5 unit2
    x_ = x
    x = qconv(x_, conv_layers[43], SW[43], SA[43])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[44], SW[44], SA[44])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[45], SW[45], SA[45])
    x_ = x + x_

    # stage5 unit3
    x = qconv(x_, conv_layers[46], SW[46], SA[46])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[47], SW[47], SA[47])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[48], SW[48], SA[48])
    x = x + x_

    # stage5 unit4
    x = qconv(x, conv_layers[49], SW[49], SA[49])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[50], SW[50], SA[50])
    x = nn.ReLU6(inplace=True)(x)
    x = qconv(x, conv_layers[51], SW[51], SA[51])

    # final_block
    x = qconv(x, conv_layers[52], SW[52], SA[52])
    x = nn.ReLU6(inplace=True)(x)
    # final_pool
    x = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)(x)
    # output
    x = qconv(x, conv_layers[53], SW[53], SA[53])
    x = x.view(x.size(0), -1)

    return x

def tait(layer1, layer2, r, t):
    assert(layer1.weight.size(0) == layer2.weight.size(1))
    for i in range(layer1.weight.size(0)):
        layer1.weight[i] = layer1.weight[i]*math.pow(6/max(r[i][0],1e-04), t)
        layer1.bias[i] = layer1.bias[i]*math.pow(6/max(r[i][0],1e-04), t)
        layer2.weight[:,i] = layer2.weight[:,i]/math.pow(6/max(r[i][0],1e-04), t)

def tait_residual(layer1, layer2, layer3, layer4, r, t):
    assert(layer3.weight.size(0) == layer4.weight.size(1))
    for i in range(layer4.weight.size(1)):
        layer1.weight[i] = layer1.weight[i]/math.pow(r[i][0], t)
        layer1.bias[i] = layer1.bias[i]/math.pow(r[i][0], t)
        layer2.weight[:,i] = layer2.weight[:,i]*math.pow(r[i][0], t)
        layer3.weight[i] = layer3.weight[i]/math.pow(r[i][0], t)
        layer3.bias[i] = layer3.bias[i]/math.pow(r[i][0], t)
        layer4.weight[:,i] = layer4.weight[:,i]*math.pow(r[i][0], t)

def run():        
    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    model = mobilenetv2_w1(pretrained=True).cuda()
    model.eval()

    # Load validation data
    #imagenet
    if args.dataset == 'imagenet': #imagenet
        test_loader = getTestData(args.dataset,
                                batch_size=args.test_batch_size,
                                path='F:\\imagenet\\',
                                for_inception=False)

        calibration_loader = getSelfBuiltCalibrationData(args.dataset, #getCalibrationData(args.dataset,
                                batch_size=args.test_batch_size,
                                path='F:\\imagenet\\',
                                for_inception=False)
    elif args.dataset == 'dog_cat': #fire dataset
        test_loader = getTestData(args.dataset,
                                batch_size=args.test_batch_size,
                                path='E:\\datasets\\archive\\dog vs cat\\dataset\\',
                                for_inception=False)
        calibration_loader = test_loader
    elif args.dataset == 'fire': #fire dataset
        test_loader = getTestData(args.dataset,
                                batch_size=args.test_batch_size,
                                path='E:\\datasets\\fire-dataset-dunnings\\images-224x224\\',
                                for_inception=False)
        calibration_loader = test_loader
        

    # Test the final quantized model
    conv_layers = []
    bn_layers = []
    for module in model.named_modules():
        if isinstance(module[1], nn.Conv2d):
            conv_layers.append(module[1])
        if isinstance(module[1], nn.BatchNorm2d):
            bn_layers.append(module[1])
        if isinstance(module[1], nn.Linear):
            fc_layer = module[1]
    fuse_bn(conv_layers, bn_layers)

    Rs = []
    QReLUs = []
    for i in range(len(conv_layers)-1):
        r = torch.zeros((conv_layers[i].out_channels, 2))
        qrelu = torch.ones(conv_layers[i].out_channels)*6
        Rs.append(r)
        QReLUs.append(qrelu)

        
    if args.dataset == 'imagenet':
        bar = Bar('Calibration', max=len(calibration_loader))
        total, correct = 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(calibration_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = calibration(inputs, conv_layers, Rs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = correct / total

                bar.suffix = f'({batch_idx + 1}/{len(calibration_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
                bar.next()
    
    #test on dogs and cats
    if args.dataset == 'dog_cat':
        bar = Bar('Calibration', max=len(calibration_loader))
        total, correct = 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(calibration_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = calibration(inputs, conv_layers, Rs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                #correct += predicted.eq(targets).sum().item()
                for indx, pred_label in enumerate(predicted):
                    if pred_label >= 158 and pred_label <= 268 and targets[indx] == 1: #cat label
                        correct += 1
                    elif pred_label >= 281 and pred_label <= 287 and targets[indx] == 0: #dog label
                        correct += 1
                acc = correct / total

                bar.suffix = f'({batch_idx + 1}/{len(calibration_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
                bar.next()
        print('\n original model Final acc: %.2f%% (%d/%d)' % (acc*100, correct, total))
            
    #test mobilenet without quantization
    '''
    bar = Bar('test on original models', max=len(test_loader))
    total, correct = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = calibration(inputs, conv_layers, Rs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct / total

            bar.suffix = f'({batch_idx + 1}/{len(test_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
            bar.next()
        print('\n original model Final acc: %.2f%% (%d/%d)' % (acc*100, correct, total))
    '''

    tait(conv_layers[0], conv_layers[1], Rs[0], args.t)
    tait(conv_layers[2], conv_layers[3], Rs[2], args.t)
    tait(conv_layers[3], conv_layers[4], Rs[3], args.t)#option
    tait(conv_layers[5], conv_layers[6], Rs[5], args.t)
    tait(conv_layers[8], conv_layers[9], Rs[8], args.t)
    tait_residual(conv_layers[6], conv_layers[7], conv_layers[9], conv_layers[10], Rs[9], args.t)
    tait(conv_layers[11], conv_layers[12], Rs[11], args.t)
    tait(conv_layers[14], conv_layers[15], Rs[14], args.t)
    tait(conv_layers[17], conv_layers[18], Rs[17], args.t)
    tait(conv_layers[20], conv_layers[21], Rs[20], args.t)
    tait(conv_layers[23], conv_layers[24], Rs[23], args.t)
    tait(conv_layers[26], conv_layers[27], Rs[26], args.t)
    tait(conv_layers[29], conv_layers[30], Rs[29], args.t)
    tait(conv_layers[32], conv_layers[33], Rs[32], args.t)
    tait(conv_layers[35], conv_layers[36], Rs[35], args.t)
    tait(conv_layers[38], conv_layers[39], Rs[38], args.t)
    tait(conv_layers[41], conv_layers[42], Rs[41], args.t)
    tait(conv_layers[44], conv_layers[45], Rs[44], args.t)
    tait(conv_layers[47], conv_layers[48], Rs[47], args.t)
    tait(conv_layers[50], conv_layers[51], Rs[50], args.t)

    if args.dataset == 'imagenet':
        bar = Bar('Calibration', max=len(calibration_loader))
        total, correct = 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(calibration_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = calibration(inputs, conv_layers, Rs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = correct / total

                bar.suffix = f'({batch_idx + 1}/{len(calibration_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
                bar.next()

    r = torch.zeros((3, 2))
    r[0,0] = (1-0.485)/0.229
    r[0,1] = (0-0.485)/0.229
    r[1,0] = (1-0.456)/0.224
    r[1,1] = (0-0.456)/0.224
    r[2,0] = (1-0.406)/0.225
    r[2,1] = (0-0.406)/0.225
    Rs.insert(0, r)

    SW, SA = [], []
    for i in range(len(conv_layers)-1):
        sw = quantize_weight(conv_layers[i].weight, args.nw)
        SW.append(sw)
        if(conv_layers[i+1].groups>1):
            next_dw = True
        else:
            next_dw = False
        sa = quantize_activation(Rs[i], next_dw, args.na)
        SA.append(sa)

    sw = quantize_weight(conv_layers[-1].weight, args.nw)
    SW.append(sw)
    sa = quantize_activation(Rs[i], False, args.na)
    SA.append(sa)

    if args.dataset == 'imagenet':
        bar = Bar('Testing', max=len(test_loader))
        total, correct, top5 = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                #outputs = qinference(inputs, conv_layers, QReLUs)
                outputs = qinference(inputs, conv_layers, SW, SA)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = correct / total


                '''         
                #imagenet
                targets = targets.cpu().numpy()
                size_of_batch = outputs.size(1)
                for count, prob in enumerate(outputs):
                    prob = prob.cpu().numpy()
                    for i in range(size_of_batch - 5, size_of_batch):
                        if np.argsort(prob)[i] == targets[count]:
                            top5 += 1

                acc_top5 = top5 / total
                '''

                #bar.suffix = f'({batch_idx + 1}/{len(test_loader)}) | ETA: {bar.eta_td} | top1: {acc} top5: {acc_top5}'

                bar.suffix = f'({batch_idx + 1}/{len(test_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
                bar.next()
        #print('\nFinal acc: %.2f%% top5 acc: %.2f%% (%d/%d)' % (acc*100, acc_top5*100, correct, total))
        print('\nFinal acc: %.2f%%  (%d/%d)' % (acc*100, correct, total))
    
    
    
    
    
    
    if args.dataset == 'dog_cat':
        bar = Bar('Testing', max=len(test_loader))
        total, correct = 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(calibration_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = qinference(inputs, conv_layers, SW, SA)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                #correct += predicted.eq(targets).sum().item()
                for indx, pred_label in enumerate(predicted):
                    if pred_label >= 158 and pred_label <= 268 and targets[indx] == 1: #cat label
                        correct += 1
                    elif pred_label >= 281 and pred_label <= 287 and targets[indx] == 0: #dog label
                        correct += 1
                acc = correct / total

                bar.suffix = f'({batch_idx + 1}/{len(calibration_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
                bar.next()
        print('\n Atfter quantization Final acc: %.2f%% (%d/%d)' % (acc*100, correct, total))
        
        
        
    bar.finish()


if __name__ == '__main__':
    #torch.multiprocessing.freeze_support()
    run()