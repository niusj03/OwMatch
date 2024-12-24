# [NeurIPS 2024] OwMatch


This is the official repository for the paper [OwMatch: Conditional Self-Labeling with Consistency for Open-World Semi-Supervised Learning](https://arxiv.org/abs/2411.01833). The repository contains the source code implemented in PyTorch.

We sincerely apologize for the delayed upload. The original server was attacked, resulting in the loss of the code. The code has been re-written and re-uploaded. If you encounter any issues or have questions, please feel free to open an issue or contact us directly.


## Dependencies
The code is built with following libraries:

* Pytorch == 2.1.0
* sklearn == 1.30

## Implementation

We employ SimCLR for pretraining. The pretrained weights can be downloaded in this [link](https://drive.google.com/file/d/19tvqJYjqyo9rktr3ULTp_E33IqqPew0D/view). Please save the weights under called `pretrained` directory.

To train on different datasets, execute the scripts located in the `fold_sh` folder. Use the following syntax:
```{bash}
bash fold_sh/<SCRIPT> owmatch.py <NAME> <LABEL RATIO> <GPU>
```
* SCRIPT: The specific script to run (e.g., dataset-specific configuration).
* NAME: The name for the experiment or run.
* LABEL RATIO: The label-to-training data ratio.
* GPU: The GPU ID to use for training.



## Acknowledgement

We gratefully acknowledge the support of the following resources:
* [ORCA](https://github.com/snap-stanford/orca)
* [TRSSL](https://github.com/nayeemrizve/TRSSL)