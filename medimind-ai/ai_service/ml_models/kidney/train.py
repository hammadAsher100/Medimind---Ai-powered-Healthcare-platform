from ml_models.train_common import run_training


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.data, "kidney"), indent=2))
