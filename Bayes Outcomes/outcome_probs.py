import pandas as pd, numpy as np, polars as pl

from get_data import read_pbp_year 


def read_data(years: list[int], data_dir: str = "Data") -> pd.DataFrame:
    """Read multiple years of PBP data and concatenate into a single DataFrame."""
    dfs = []
    for year in years:
        df_year = read_pbp_year(year, data_dir=data_dir)
        dfs.append(df_year)
    batter_stats = pd.read_parquet("Data/batter_stats.parquet")
    pitcher_stats = pd.read_parquet("Data/pitcher_stats.parquet")
    return pd.concat(dfs, ignore_index=True), batter_stats, pitcher_stats


def add_swing_length(pbp: pd.DataFrame, batter_stats: pd.DataFrame) -> pd.DataFrame:
    sl = (
        pbp.dropna(subset=["swing_length"])
        .groupby(["batter", "game_year"], as_index=False)
        .agg(swing_length=("swing_length", "mean"))
        .rename(columns={"batter": "xMLBAMID", "game_year": "Season"})
    )

    batter_stats = batter_stats.merge(                                                    
      sl,                                                                               
      on=["xMLBAMID", "Season"],                                             
      how="left"                                                                        
  )  
    return batter_stats

df, batter_stats, pitcher_stats = read_data(years=[2024, 2025])
batter_stats = add_swing_length(df, batter_stats)

