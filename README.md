<<<<<<< HEAD
# neural-network-project
=======
# TSMixer in PyTorch

The implementation is done based on the 2 .yml files, respectively conf.etydataset.yml and conf.etydataset.gridsearch.yml where are specified all the parameters to train the models.



## Sample results

![Predictions on validation set](readme_figures/preds.png)
*Predictions on validation set*

![Training loss](readme_figures/loss.png)
*Loss during training*

Parameters used for example:
* `input_length`: 512
* `prediction_length`: 96
* `no_features`: 7
* `no_mixer_layers`: 4
* `dataset`: ETTh1.csv
* `batch_size`: 32
* `num_epochs`: 100 with early stopping after 5 epochs without improvement
* `learning_rate`: 0.00001
* `optimizer`: Adam
* `validation_split_holdout`: 0.2 - last 20% of the time series data is used for validation
* `dropout`: 0.3
* `feat_mixing_hidden_channels`: 256 - number of hidden channels in the feature mixing layer

## Data

You can find the raw ETDataset data [here](https://github.com/zhouhaoyi/ETDataset/tree/11ab373cf9c9f5be7698e219a5a170e1b1c8a930)


## Running

Install the requirements:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python main.py --conf  coeficients.yml --command train

Instead you can change coeficients.yml with best.coeficients.yml for better accuracy but more time to train
```

The output will be in the `output_dir` directory specified in the config file. The config file is in YAML format. The format is defined by [utils/tsmixer_conf.py](utils/tsmixer_conf.py).

Plot the loss curves:

```bash
python main.py --conf coeficients.yml --command loss --show
```

Predict some of the validation data and plot it:

```bash
python main.py --conf coeficients.yml --command predict --show
```

Run a grid search over the hyperparameters:

```bash
python main.py --conf gridsearch.coeficients.yml --command grid-search
```

Note that the format of the config file is different for the grid search. The format is defined by [utils/tsmixer_grid_search_conf.py](utils/tsmixer_grid_search_conf.py).

>>>>>>> d73bd101d (Final project upload)
