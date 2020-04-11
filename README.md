# NLP_project
ENSAE NLP project under the supervision of Benjamin Muller of Etienne Sullice and Charles Laroche.

Colab notebook: [our colab](https://colab.research.google.com/drive/1IXUwZ3S1X47GyrcsRdneCk92-kTUZTR_)

## Code structure

```
NLP_Project
    ├── beam_s.py
        └── caption_image_beam_search (function)
    ├── dataset.py
        ├── EncodeCaption (Class)
        └── CocoDataset (Class)
    ├── networks.py
        ├── Encoder (Class)
        ├── Attention (Class)
        └── DecoderWithAttention (Class)
    ├── visualize.py
        └── visualize_att (function)
    └── utils.py
        ├── plot_torch_img (function)
        ├── opposite_split (function)
        ├── AverageMeter (Class) 
        ├── adjust_learning_rate (function)
        ├── accuracy (function)
        ├── save_checkpoint (function)
        └── clip_gradient (function)
```
- **beam_s.py** contains the code to predict a sentence using beam search
- **dataset.py** contains our caption encoder and the dataset class that will handle our data and pre-processed it.
- **networks.py** contains all the network architecture, the ResNet 101 encoder, Attention network and an Attention based decoder.
- **visualize.py** contains a function that do visualization of images with attention coefficients.
- **utils.py** contains some utilities

## References

[Long short-term memory](https://www.bioinf.jku.at/publications/older/2604.pdf)<br>
Sepp Hochreiter, Jürgen Schmidhuber.

[An Introductory Survey on Attention Mechanisms in NLP Problems](https://arxiv.org/pdf/1811.05544.pdf)<br>
Dichao Hu}.
  In ICCV 2017. (* equal contributions)

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)<br>
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. EfrosKaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. 
  
[BLUE: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf) <br>
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu.
