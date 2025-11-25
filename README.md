### Title: 
A Weakly-Supervised Contrastive Learning Framework for Few-Shot Code Readability Classification (WCR-CLC)

### Introduction: 
This project provide the code of WCR-CLC.

### Installation:  
python 3.6

Bert

### Usage:  
Please read our article first to understand the code process as well as the datasets corresponding to pre-training and fine-tuning.

For the token-based backbone network, we first run the token-based_pre.py for pre-training and save the generated model weights in the s.h5. Then, we run fine-texture.py for fine-tuning to obtain the results. During this period, please make sure to use the correct dataset.

For the character-based backbone network, the overall process is consistent with the above, with model weights saved in the t.h5. However, it is necessary to change the preprocess_structure_data() function and adjust the representation dimensions in the triplet_model_stru.py and fine_stru.py according to the dataset used during training.