# LLLR: Toy-model
A simple 3-layer fully connected network performing the density ratio estimation using the loss for log-likelihood ratio estimation (LLLR).

The structure of this project is inherited from [the SPRT-TANDEM code](https://github.com/TaikiMiyagawa/SPRT-TANDEM).  

## Tested Environment
- Python 3.5
- tensorflow 2.0.0
- CUDA 10.0
- cuDNN 7.6.4.38

## Tutorial 

## Example Results
The MLP was trained either with the LLLR or CE-loss, repeated 56 times with different random initial vairables. The plot below shows the mean NMSE with the shading shows standard error of the mean.
![](./example_results/LLLRvsCE_NMSE.png)

## Reference
[1] Sugiyama, M.; Suzuki, T.; Nakajima, S.; Kashima, H.; von Bünau, P.; Kawanabe, M. Direct Importance Estimation for Covariate Shift Adaptation. Ann Inst Stat Math 2008, 60 (4), 699–746. https://doi.org/10.1007/s10463-008-0197-x.

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
