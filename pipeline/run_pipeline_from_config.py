from utils.data_utils import load_dataset_bundle, check_duplicate
from utils.config import load_train_config, load_visual_config
from utils.model_utils import initialize_model_and_config
from utils.visual_utils import plot_loss_curves, generate_all_regression_plots

from pipeline.preprocess import DataPreprocessor
from pipeline.trainer import ModelTrainer
from pipeline.result_saver import save_all_evaluation_outputs


FEATURES_PATH = './data/fingerprints/AGILE'
LABELS_PATH = './data/AGILE.csv'


def run_pipeline_from_config(config_path):

    train_config = load_train_config(configpath=config_path)
    visual_config = load_visual_config()

    features_type = train_config['features']
    model_config = train_config['model_config']
    model_name = model_config['model']

    print(f"Running pipeline with {', '.join(features_type)} features and {model_name} model")

    print("Loading saved data...")
    features, labels, idxs, num_features = load_dataset_bundle(
        features_path=FEATURES_PATH,
        labels_path=LABELS_PATH,
        feature_names=features_type
    )

    print("Checking for duplicate entries...")
    check_duplicate(features)

    print("Preprocessing and splitting data...")
    preprocessor = DataPreprocessor(labels, features, idxs)
    split = preprocessor.split(train_config["split"], train_config["dataset"])

    print(f"Initializing model '{train_config['model_config']['model']}' with input size {num_features}...")
    model, config = initialize_model_and_config(train_config, inp_size=num_features)

    trainer = ModelTrainer(model, config, preprocessor.scaler,
                        train_data=(split["X"][0], split["y"][0]),
                        val_data=(split["X"][1], split["y"][1]),
                        test_data=(split["X"][2], split["y"][2]))

    print("Training model...")
    loss_dict = trainer.train()
    trainer.save_best_model()

    print("Evaluating model...")
    train_preds, valid_preds, test_preds = trainer.evaluate()

    print("Saving results...")
    save_all_evaluation_outputs(
        train_preds, valid_preds, test_preds,
        split["true"][0], split["true"][1], split["true"][2],
        split["idxs"][0], split["idxs"][1], split["idxs"][2],
        config
    )

    print("Saving plots...")
    plot_loss_curves(loss_dict, config, save=True)
    generate_all_regression_plots(
    train_preds, valid_preds, test_preds,
    split["true"][0], split["true"][1], split["true"][2],
    config
    )
