# On the Ramifications of Human Label Uncertainty

---
This work was mainly done in the [Omni Lab for Intelligent Visual Engineering and Science (OLIVES)](https://ghassanalregib.info/) @ Georgia Tech.
Feel free to check our lab's [Website](https://ghassanalregib.info/) and [GitHub](https://github.com/olivesgatech) for other interesting work!!!
[<img align="right" src="https://www.dropbox.com/s/rowej0iof65fie5/OLIVES-logo_with_website.png?raw=1" width="15%">](https://ghassanalregib.info/)

---
C. Zhou, M. Prabhushankar, and G. AlRegib, "On the Ramifications of Human Label Uncertainty," in *NeurIPS 2022 Workshop on Human in the Loop Learning*, New Orleans, LA, Nov. 28 - Dec. 9 2022.

<p align="center">
<img src="https://drive.google.com/uc?export=download&id=1ngoLaSlHEnxzr2dTlT1wBbSsklbfbqZB" width="80%">
</p>

### Abstract
Humans exhibit disagreement during data labeling. We term this disagreement as human label uncertainty. In this work, we study the ramifications of human label uncertainty (HLU). Our evaluation of existing uncertainty estimation algorithms, with the presence of HLU, indicates the limitations of existing uncertainty metrics and algorithms themselves in response to HLU. Meanwhile, we observe undue effects in predictive uncertainty and generalizability. To mitigate the undue effects, we introduce a novel natural scene statistics (NSS) based label dilution training scheme without requiring massive human labels. Specifically, we first select a subset of samples with low perceptual quality ranked by statistical regularities of images. We then assign separate labels to each sample in this subset to obtain a training set with diluted labels. Our experiments and analysis demonstrate that training with NSS-based label dilution alleviates the undue effects caused by HLU.

<p align="center">
<img src="https://drive.google.com/uc?export=download&id=1JqCvfXgkBoat19plZGcHIYwtdx1bW8Ci" width="80%">
</p>

---
## Getting Started

See the [Colab Notebook](https://colab.research.google.com/drive/1xo0bOQbOfQ_0GjEcDWPSZkkvl7_p84g3?usp=sharing)
to learn about basic usage.

---
### Usage
Training/validation using our NSS-based label dilution: Run the following command in a bash script.
```
for (( i=0; i<=4; ++i ))
do
python3 core/main.py --config_file configs/uncertain_quant.yml \
                     --gpu 0 \
                     ckpt_dir <savepath>/label_uncertainty/ \
                     trainset.setting MultiLabel \
                     trainset.nlbl.type nss \
                     trainset.nlbl.rate 0.4021 \
                     trainset.nlbl.rate_aggre 0.0903 \
                     trainset.unclbl.assign.name kmeans \
                     trainset.unclbl.assign.noise_type clean \
                     trainset.unclbl.nss brisque \
                     trainset.noise_human True \
                     solver.batch_size 128 \
                     solver.num_epoch 200 \
                     uncertain_quant.method ordinary \
                     uncertain_quant.backbone.arch resnet \
                     uncertain_quant.backbone.depth 18 \
                     seed "$i"

done
```
---
### Acknowledgments
This work was mainly done in [OLIVES@GT](https://ghassanalregib.info/) with the guidance of Prof. [Ghassan AlRegib](https://www.linkedin.com/in/ghassan-alregib-0602131), and the collaboration with Dr. [Mohit Prabhushankar](https://www.linkedin.com/in/mohitps). 

---
### Contact
[Chen Zhou](https://www.linkedin.com/in/czhou88) <br>
chen DOT zhou AT gatech DOT edu <br>
[<img align="left" src="https://www.dropbox.com/s/rowej0iof65fie5/OLIVES-logo_with_website.png?raw=1" width="15%">](https://ghassanalregib.info/)
