# EMPIRICAL-ASSET-PRICING-VIA-THE-CONDITIONAL-QUANTILE-VARIATIONAL-AUTOENCODER
This is a pre-release version of code for:
# Asset pricing via the conditional quantile variational autoencoder
This resposity is a pre-released verison of Python code used in the paper "[Asset pricing via the conditional quantile variational autoencoder](https://www.researchgate.net/publication/361455269_Asset_pricing_via_the_conditional_quantile_variational_autoencoder)".

## Reproduce the simulation results with certain hyperparameters
```
# CAE
python main.py --model CAE --save-folder logs/CAE

# CVAE
python main.py --model CVAE --save-folder logs/CVAE --lam 0

# CQAE
python main.py --model CVAE --save-folder logs/CVAE --lam 2e-5

# CQVAE
python main.py --model CVAE --save-folder logs/CVAE --lam 0
```

## Reproduce the empirical analysis result
1. Download data from the homepage of Prof. Dacheng Xiu [data link](https://dachxiu.chicagobooth.edu/download/datashare.zip)
2. Preprocess the data with the code in [code link](https://feng-cityuhk.github.io/EquityCharacteristics/)
3. Feed the processed data into our algorithm.
See [main.py]() for more details.

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
