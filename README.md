# Probability density ratio estimation with the Loss for Log-Likelihood Ratio
This article supplements our paper, [Deep Neural Networks for the Sequential Probability Ratio Test on Non-i.i.d. Data Series](https://arxiv.org/abs/2006.05587).  

Below we test whether the proposed Loss for Log-Likelihood Ratio Loss (LLLR) can help a neural network estimating the true probability density ratio. Providing the ground-truth probability density ratio was difficult in the original paper because it was prohibitive to find the true probability distribution out of the public databases containing real-world scenes. Thus, we create a toy-model estimating the probability density ratio of the two multivariate Gaussian distributions. Experimental results show that a multi-layer perceptron (MLP) trained with the proposed LLLR achieves smaller estimation error than an MLP with crossentropy (CE)-loss.

## Requirements
This article is best read with the Chrome browser with [MathJax Plugin for GitHub](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en).

## Experimental Settings
Following Sugiyama et al. 2008 [1], let $p_0(x)$ be the $d$-dimensional Gaussian density with mean $(2, 0, 0, ..., 0)$ and covariance identity, and $p_1(x)$ be the $d$-dimensional Gaussian density with mean $(0, 2, 0, ..., 0)$ and covariance identity. 

The task for the neural network is to estimate the density ratio:

\begin{equation}
\hat{r}(x_i) = \frac{\hat{p}_1(x_i)}{\hat{p}_0(x_i)}.
\end{equation}

Here, $x$ is sampled from one of the two Gaussian distributions, $p_0$ or $p_1$, and is associated with class label $y=0$ or $y=1$, respectively. We compared 2 loss functions, CE-loss and LLLR:

\begin{equation}
\mathrm{LLLR}:= \frac{1}{N}\sum_{i=1}^{N}
\left|
    y - \sigma\left(\log\hat{r_i}\right)
\right|
\end{equation}

where $\sigma$ is the sigmoid function.

A simple Neural network consists of 3-layer fully-connected network with nonlinear activation (ReLU) is used for estimating $\hat{r}(x)$.  

Evaluation metric is normalized mean squared error (NMSE, [1]):

\begin{equation}
\mathrm{NMSE}:= \frac{1}{N}\sum_{i=1}^{N}
\left(
    \frac{\hat{r_j}}{\sum_{j=1}^{N}\hat{r_j}} -
    \frac{r_i}{\sum_{j=1}^{N}r_j}
\right)
\end{equation}


## Tested Environment
- Python 3.5
- tensorflow 2.0.0
- CUDA 10.0
- cuDNN 7.6.4.38

## Tutorial 
The structure of this repo is inherited from [the original SPRT-TANDEM code](https://github.com/TaikiMiyagawa/SPRT-TANDEM). For the details, see the README of the original code.
- To train the MLP, use train_MLP.py. To change parameters including weights for the LLLE and CE-loss, modify .yaml files under the config folder.  

- To visualize the example results, use example_results/plot_example_runs.ipynb. Also see the plot below.

## Example Results
The MLP was trained either with the LLLR or CE-loss, repeated 56 times with different random initial vairables. The plot below shows the mean NMSE with the shading shows standard error of the mean.
![](./example_results/LLLRvsCE_NMSE.png)

## Reference
[1] Sugiyama, M.; Suzuki, T.; Nakajima, S.; Kashima, H.; von Bünau, P.; Kawanabe, M. Direct Importance Estimation for Covariate Shift Adaptation. Ann Inst Stat Math 2008, 60 (4), 699–746.

## Citation
___Please cite our paper if you use the whole or a part of our codes.___
```
Bibtex:

@misc{SPRT_TANDEM2020,
    title={Deep Neural Networks for the Sequential Probability Ratio Test on Non-i.i.d. Data Series},
    author={Akinori F. Ebihara and Taiki Miyagawa and Kazuyuki Sakurai and Hitoshi Imaoka},
    year={2020},
    eprint={2006.05587},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

```
