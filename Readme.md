This project code is implementing quantization methodologies as well as network pruning to achieve compression on MobileNetV2.
The pruning parts are the modifications from the code provided by wlguan et al., which is available at https://github.com/wlguan/MobileNet-v2-pruning


##########################################################################################
if you want to test pure quantization results on imagenet:

    ```
    python tait.py --dataset imagenet --t 0.3
    (note: you can change your tait value from range of [0, 1] for testing different cases)
    ```
##########################################################################################
if you want to test network pruning approaches on Cifar-10 datasets:
1. Training
    ```
   python main.py --arch MobileNetV2 (for l1norm pruner )
   python main.py --sr --arch MobileNetV2 (for slimming pruner) 
   python main.py --arch USMobileNetV2 (for Autoslim pruner )
    ```
2. Pruning (prune+finetune)
    ```
   python prune.py --arch MobileNetV2 --pruner l1normPruner --pruneratio 0.6
   python prune.py --arch MobileNetV2 --pruner SlimmingPruner --sr --pruneratio 0.6
   python prune.py --arch USMobileNetV2 --pruner AutoSlimPruner
    ```
(note: the model should be trained first, and each pruning strategies require respective training methods,
the prune ratio could be self-defined to test various case)

##########################################################################################
if you want to test network pruning approaches on ImageNet datasets:
    ````
   python prune_imagenet.py --arch MobileNetV2 --pruner l1normPruner --pruneratio 0.6
   python prune_imagenet.py --arch MobileNetV2 --pruner SlimmingPruner --sr --pruneratio 0.6

(note: prune ratio could alter)
    ````
##########################################################################################
if you want to test combinations of network pruning and quanatization approaches on ImageNet datasets:
    ````
   python tait_pruning.py --arch MobileNetV2 --pruner l1normPruner --sr --pruneratio 0.3 --t 0.3
   python tait_pruning.py --arch MobileNetV2 --pruner SlimmingPruner --sr --pruneratio 0.3 --t 0.3
(note: prune ratio, tait ratio could alter)
    ````
##########################################################################################
to have comprehensive testing on various variables, please read contents of evaluations to modify it, then run:
    ````
   python evaluation.py
    ````

## Reference
[rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning) 

[Pruned-MobileNet_v2](https://github.com/eezywu/Pruned-MobileNet_v2) 

[MobileNet-v2-pruning](https://github.com/wlguan/MobileNet-v2-pruning)
