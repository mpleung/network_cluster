This repository contains replication files for "Network Cluster-Robust Inference."

Coded for Python 3.9.7. Packages used: numpy (v1.20.3), pandas (v1.3.4), scipy (v1.7.1), networkx (v2.6.3), sklearn (v0.24.2), matplotlib (v3.4.3), seaborn (v0.11.2).

The following can be done in any order.

* Run `spectrum.py` with `make_plot=True` to produce Figure 4 and images for Figure 1. Rerun with `make_plot=False` to produce Table 1.

* Run `design1_size.py` to produce Table 2. Run `design1_power.py` and then `plot_power_design1.py` to produce Figure 5.

* Download the Paluck et al. (2016) data from [ICPSR](https://www.icpsr.umich.edu/web/civicleads/studies/37070), extract the contents, and then place the data file `37070-0001-Data.tsv` into this directory. Run `design2.py` three times, one for each of `network_model = 'RGG'`, `'RCM'`, `'config'`. Then run `tables_design2.py` to produce Table 3.

    On the server I used, for `RGG` and `RCM`, `design2.py` completes in less than 24 hours with 32 cores (change the `processes` argument in the file to set number of cores). For `config`, I separated into two runs with `num_schools=[1,2]` and then `num_schools=[4]`, with each completing in less than 24 hours.

* Download the Zacchia (2020) data from the publisher's [website](https://academic.oup.com/restud/article-abstract/87/4/1989/5505452?redirectedFrom=fulltext). Extract contents into a folder in this directory called 'zacchia'. Run `application.py` to produce Figure 2 and 3.
