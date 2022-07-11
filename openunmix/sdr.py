import museval
import argparse

def main():
    parser = argparse.ArgumentParser(description="MUSDB18 Evaluation", add_help=False)

    parser.add_argument(
        "--model",
        type=str,
        help="give model name to be evaluated",
    )

    args = parser.parse_args()

    method = museval.MethodStore()
    method.load('models/' + args.model + '.pandas')
    method.agg_frames_tracks_scores()