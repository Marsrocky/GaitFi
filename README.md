# GaitFi: Robust Device-Free Human Identification via WiFi and Vision Multimodal Learning

## Introduction
As an important biomarker for human identification, human gait can be collected at a distance by passive sensors without subject cooperation, which plays an essential role in crime prevention, security detection and other human identification applications. At present, most research works are based on cameras and computer vision techniques to perform gait recognition. However, vision-based methods are not reliable when confronting poor illuminations, leading to degrading performances. In this paper, we propose a novel multimodal gait recognition method, namely GaitFi, which leverages WiFi signals and videos for human identification. In GaitFi, Channel State Information (CSI) that reflects the multi-path propagation of WiFi is collected to capture human gaits, while videos are captured by cameras. To learn robust gait information, we propose a Lightweight Residual Convolution Network (LRCN) as the backbone network, and further propose the two-stream GaitFi by integrating WiFi and vision features for the gait retrieval task. The GaitFi is trained by the triplet loss and classification loss on different levels of features. Extensive experiments are conducted in the real world, which demonstrates that the GaitFi outperforms state-of-the-art gait recognition methods based on single WiFi or camera, achieving 94.2\% for human identification tasks of 12 subjects.


## Requirements

```
scipy - 1.5.4
numpy - 1.21.5
torchvision - 0.11.2
pytorch - 1.7.0
```



## Training
Train using vision modality only: python train.py --input_type image

Train using WiFi modality only: python train.py --input_type mat

Train using both vision and WiFi modality: python train.py --input_type both

## Testing
Copy the model saved to save_models in training to best_models

Test using vision modality only: python test.py --input_type image

Test using WiFi modality only: python test.py --input_type mat

Test using WiFi modality only: python test.py --input_type both



## Model

The GaitFi has the following components:

- ***class*** **CNN** : LRCN block
- ***class*** **RNN** : LSTM block
- ***class*** **CRNN**: Fusion mechanism of WiFi and vision modalities

## Reference

```
@ARTICLE{9887951,
  author={Deng, Lang and Yang, Jianfei and Yuan, Shenghai and Zou, Han and Lu, Chris Xiaoxuan and Xie, Lihua},
  journal={IEEE Internet of Things Journal}, 
  title={GaitFi: Robust Device-Free Human Identification via WiFi and Vision Multimodal Learning}, 
  year={2022},
  publisher={IEEE},
  doi={10.1109/JIOT.2022.3203559}}
```