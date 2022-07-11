import museval
import argparse

def get_scores(model_name):
    method = museval.MethodStore()
    method.load('models/' + model_name + '.pandas')
    method.agg_frames_tracks_scores()
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUSDB18 Evaluation", add_help=False)

    parser.add_argument(
        "--model",
        type=str,
        help="give model name to be evaluated",
    )

    args = parser.parse_args()

    get_scores(args.model)