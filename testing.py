import pandas as pd
import museval
import matplotlib.pyplot as plt

eval_df = museval.MethodStore()
eval_df.load("baseline.pandas")
df = eval_df.agg_frames_scores()
# Get Sdr score for each unique track
df = df.to_frame()
sdr_df = df.iloc[df.index.get_level_values('metric') == "SDR" ]
vocals_sdr_df = sdr_df.loc[sdr_df.index.get_level_values('target') == "vocals"]
vocals_sdr_df = vocals_sdr_df.sort_values(by="track", ascending=False).droplevel(["method", "target", "metric"])
vocals_sdr_df = vocals_sdr_df.reset_index()
# print(vocals_sdr_df)
# vocals_sdr_df = vocals_sdr_df.drop(columns=["track"])
print(vocals_sdr_df)
vocals_sdr_df = vocals_sdr_df.rename(columns={"score": "Baseline"})
# vocals_sdr_df.plot()

eval_df_transformer = museval.MethodStore()
eval_df_transformer.load("transformer.pandas")
df_transformer = eval_df_transformer.agg_frames_scores()
# Get Sdr score for each unique track
df_transformer = df_transformer.to_frame()
sdr_df_transformer = df_transformer.iloc[df_transformer.index.get_level_values('metric') == "SDR" ]
vocals_sdr_df_transformer = sdr_df_transformer.loc[sdr_df_transformer.index.get_level_values('target') == "vocals"]
vocals_sdr_df_transformer = vocals_sdr_df_transformer.sort_values(by="track", ascending=False).droplevel(["method", "target", "metric"])
vocals_sdr_df_transformer = vocals_sdr_df_transformer.reset_index()
# print(vocals_sdr_df)
# vocals_sdr_df_transformer = vocals_sdr_df_transformer.drop(columns=["track"])
vocals_sdr_df_transformer = vocals_sdr_df_transformer.rename(columns={"score": "Transformer"})
print(vocals_sdr_df_transformer)
# vocals_sdr_df_transformer.plot()

vocals_sdr_scores = pd.concat([vocals_sdr_df, vocals_sdr_df_transformer["Transformer"]], axis=1)
vocals_sdr_scores.plot()


genre_df = pd.read_csv("tracklist.csv")[["Track Name", "Genre"]]
vocals_sdr_scores["genre"] = vocals_sdr_scores.apply(lambda x: genre_df[x["track"] == genre_df["Track Name"]]["Genre"].values[0], axis=1)
print(genre_df)
print(vocals_sdr_scores)

vocals_sdr_scores.plot.bar()

vocals_sdr_scores.groupby("genre").mean().plot.bar(ylabel="SDR (dB)", xlabel="Genre", title="Baseline and Transformer Performance by Genre")
plt.show()