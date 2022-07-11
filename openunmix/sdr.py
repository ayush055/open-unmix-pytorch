import museval
import argparse

def get_scores(model_name):
    method = museval.MethodStore()
    method.load('models/' + model_name + '.pandas')
    return method.agg_frames_tracks_scores()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUSDB18 Evaluation", add_help=False)

    parser.add_argument(
        "--model",
        type=str,
        help="give model name to be evaluated",
    )

    args = parser.parse_args()

    scores = get_scores(args.model)
    print(scores)