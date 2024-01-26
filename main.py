from utils import TSMixer, plot_preds, plot_loss, TSMixerConf, TSMixerGridSearch

import argparse
import yaml
import os

if __name__ == "__main__":
    # Parsing command-line arguments
    parser = argparse.ArgumentParser()
    # This is to be chosen when running the model, to chose which action to perform
    parser.add_argument("--command", type=str, required=True, choices=["train", "predict", "loss", "grid-search"], help="Command to run")
    parser.add_argument("--conf", type=str, required=False, help="Path to the configuration file")
    parser.add_argument("--no-feats-plot", type=int, required=False, default=6, help="Number of features to plot")
    parser.add_argument("--show", action="store_true", required=False, help="Show plots")
    args = parser.parse_args()
    # Handling the 'train' command
    if args.command == "train":
        # Load configuration as stated in the configuration file
        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf = TSMixerConf.from_dict(yaml.safe_load(f))
        
        # Set the device to CPU as I don't have a gpu to use in my machine
        conf.device = "cpu"
        #Create a ts mixer instance
        tsmixer = TSMixer(conf)
        # Train the TSMixer model
        tsmixer.train()
    # Handling the 'predict' command
    elif args.command == "predict":

        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf = TSMixerConf.from_dict(yaml.safe_load(f))

        # Load best checkpoint
        conf.initialize = TSMixerConf.Initialize.FROM_BEST_CHECKPOINT

        # Set the device to CPU
        conf.device = "cpu"

        tsmixer = TSMixer(conf)

        # Predict on the validation dataset
        data = tsmixer.predict_val_dataset(max_samples=10, save_inputs=False)

        # Plot predictions
        data_plt = data[0]
        assert args.no_feats_plot is not None, "Must provide the number of features to plot"
        plot_preds(
            preds=data_plt.pred,
            preds_gt=data_plt.pred_gt,
            no_feats_plot=args.no_feats_plot,
            show=args.show,
            fname_save=os.path.join(conf.image_dir, "preds.png")
        )
    # Handling the 'loss' command
    elif args.command == "loss":

        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf = TSMixerConf.from_dict(yaml.safe_load(f))

        # Set the device to CPU
        conf.device = "cpu"

        train_data = conf.load_training_metadata_or_new()
        plot_loss(
            train_data=train_data,
            show=args.show,
            fname_save=os.path.join(conf.image_dir, "loss.png")
        )
    # Handling the 'grid-search' command
    elif args.command == "grid-search":

        # Load configuration
        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf_grid_search = TSMixerGridSearch.from_dict(yaml.safe_load(f))

        # Run grid search
        for conf in conf_grid_search.iterate():
            # Set the device to CPU
            conf.device = "cpu"

            tsmixer = TSMixer(conf)
            tsmixer.train()

    else:
        raise NotImplementedError(f"Command {args.command} not implemented")
