import pandas as pd
import museval

eval_df = museval.MethodStore()
eval_df.load("baseline.pandas")
df = eval_df.agg_frames_scores()
# Get Sdr score for each unique track
df = df.to_frame()
sdr_df = df.iloc[df.index.get_level_values('metric') == "SDR" ]
vocals_sdr_df = sdr_df.loc[sdr_df.index.get_level_values('target') == "vocals"]
print(vocals_sdr_df.sort_values(by="score", ascending=False))

eval_df = museval.MethodStore()
eval_df.load("transformer.pandas")
df = eval_df.agg_frames_scores()
# Get Sdr score for each unique track
df = df.to_frame()
sdr_df = df.iloc[df.index.get_level_values('metric') == "SDR" ]
vocals_sdr_df = sdr_df.loc[sdr_df.index.get_level_values('target') == "vocals"]
print(vocals_sdr_df.sort_values(by="score", ascending=False))