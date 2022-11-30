# Asset pricing via the conditional quantile variational autoencoder
This resposity is a pre-released verison of Python code used in the paper "[Asset pricing via the conditional quantile variational autoencoder](https://www.researchgate.net/publication/361455269_Asset_pricing_via_the_conditional_quantile_variational_autoencoder)".

If you have any question about this resposity or implement details of paper, please feel free to submit a new issue.
 
## Reproduce the simulation results with certain hyperparameters
```
# CAE
python main.py --model CAE --save-folder logs/CAE --lam 1e-4

# CVAE
python main.py --model CVAE --save-folder logs/CVAE

# CQAE
python main.py --model CQAE --save-folder logs/CQAE --lam 2e-5

# CQVAE
python main.py --model CQVAE --save-folder logs/CQVAE
```

## Reproduce the empirical analysis result
1. Download data from the homepage of Prof. Dacheng Xiu ([homepage link](https://dachxiu.chicagobooth.edu) and [data link](https://dachxiu.chicagobooth.edu/download/datashare.zip)).
2. Preprocess the data with the code in [code link](https://feng-cityuhk.github.io/EquityCharacteristics/).
3. Feed the processed data into our algorithm.
See main.py for more details.

## Citation
Please cite our paper if you feel this code helps.
```
@article{yang2022CQVAE,
    author = {Yang, Xuanling and Zhu, Zhoufan and Li, Dong and Zhu, Ke},
    year = {2022},
    month = {06},
    pages = {},
    title = {Asset pricing via the conditional quantile variational autoencoder}
}
```
