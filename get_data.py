from pathlib import Path
from typing import Optional
import re
import unicodedata
import polars as pl
import pandas as pd
import requests
from pybaseball import statcast


PVC_BATTERS = {
    "Oneil Cruz",
    "James Wood",
    "Ryan McMahon",
    "Riley Greene",
    "Eugenio Suarez",
    "Elly De La Cruz",
    "Kyle Schwarber",
    "Jo Adell",
    "Teoscar Hernandez",
    "Spencer Torkelson",
    "Lawrence Butler",
    "Randy Arozarena",
    "Adolis Garcia",
    "Jazz Chisholm Jr.",
    "Christian Walker",
    "Michael Busch",
    "Willy Adames",
    "Taylor Ward",
    "Rafael Devers",
    "Brent Rooker",
    "Zach Neto",
    "Matt Olson",
    "Marcell Ozuna",
    "Ian Happ",
    "Pete Crow-Armstrong",
    "Pete Alonso",
    "Shea Langeliers",
}


def _normalize_player_name(name: object) -> str:
    """Normalize name text so matching is case/spacing/accent insensitive."""
    if pd.isna(name):
        return ""
    text = str(name).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


PVC_BATTER_LOOKUP = {_normalize_player_name(name) for name in PVC_BATTERS}


def get_statcast_pbp_year(year: int) -> pd.DataFrame:
    """Fetch one full year of Statcast pitch-by-pitch and keep regular season only."""
    df = statcast(start_dt=f"{year}-01-01", end_dt=f"{year}-12-31")
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"].copy()
    return df.reset_index(drop=True)


def get_players(
    season: int,
    sport_id: int = 1,
    game_type: Optional[list[str]] = None) -> pd.DataFrame:
    """Get season player table from MLB StatsAPI (ID, full name, position, team)."""
    game_type = game_type or ["R"]
    game_type_str = ",".join(str(x) for x in game_type)
    url = f"https://statsapi.mlb.com/api/v1/sports/{sport_id}/players"
    response = requests.get(
        url,
        params={"season": season, "gameType": f"[{game_type_str}]"},
        timeout=60,
    )
    response.raise_for_status()
    people = response.json().get("people", [])

    records: list[dict[str, object]] = []
    for p in people:
        records.append(
            {
                "player_id": p.get("id"),
                "full_name": p.get("fullName"),
                "position": (p.get("primaryPosition") or {}).get("abbreviation"),
                "team": (p.get("currentTeam") or {}).get("id"),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["player_id", "full_name", "position", "team"],
    )


def add_player_names(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Merge season player IDs to readable names for batter/pitcher columns."""
    if "batter" not in df.columns and "pitcher" not in df.columns:
        return df

    players = get_players(season=season, sport_id=1, game_type=["R"])
    if players.empty:
        return df

    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("Int64")
    player_lookup = (
        players[["player_id", "full_name"]]
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"])
        .rename(columns={"full_name": "player_name"})
    )

    out = df.copy()

    if "batter" in out.columns:
        out["batter"] = pd.to_numeric(out["batter"], errors="coerce").astype("Int64")
        batter_lookup = player_lookup.rename(
            columns={"player_id": "batter", "player_name": "batter_name"}
        )
        out = out.merge(batter_lookup, on="batter", how="left")

    if "pitcher" in out.columns:
        out["pitcher"] = pd.to_numeric(out["pitcher"], errors="coerce").astype("Int64")
        pitcher_lookup = player_lookup.rename(
            columns={"player_id": "pitcher", "player_name": "pitcher_name"}
        )
        out = out.merge(pitcher_lookup, on="pitcher", how="left")

    return out


def add_pvc_flag(df: pd.DataFrame, batter_col: Optional[str] = None) -> pd.DataFrame:
    """Add PVC column (1/0) for rows where batter is in the configured name list."""
    out = df.copy()

    if batter_col is None:
        if "batter_name" in out.columns:
            batter_col = "batter_name"
        elif "batter" in out.columns:
            batter_col = "batter"

    if batter_col is None or batter_col not in out.columns:
        out["PVC"] = 0
        return out

    normalized_batters = out[batter_col].map(_normalize_player_name)
    out["PVC"] = normalized_batters.isin(PVC_BATTER_LOOKUP).astype("int8")
    return out


def save_pbp_year(df: pd.DataFrame, year: int, data_dir: str = "Data") -> Path:
    """Save a yearly PBP DataFrame to Data/pbp_YYYY.parquet."""
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pbp_{year}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def read_pbp_year(year: int, data_dir: str = "Data") -> pd.DataFrame:
    """Read a yearly PBP DataFrame from Data/pbp_YYYY.parquet."""
    in_path = Path(data_dir) / f"pbp_{year}.parquet"
    return pd.read_parquet(in_path)


def scrape_and_save_statcast_pbp_years(
    start_year: int = 2021,
    end_year: int = 2025,
    data_dir: str = "Data") -> dict[str, pd.DataFrame]:
    """Scrape and save yearly regular-season Statcast PBP for a year range."""
    yearly_data: dict[str, pd.DataFrame] = {}
    for year in range(start_year, end_year + 1):
        df = get_statcast_pbp_year(year)
        df = add_player_names(df, season=year)
        df = add_pvc_flag(df)
        save_pbp_year(df, year, data_dir=data_dir)
        yearly_data[f"pbp_{year}"] = df
    return yearly_data


def apply_names_to_saved_pbp(data_dir: str = "Data") -> dict[str, Path]:
    """Read each Data/pbp_YYYY.parquet, add player names, and overwrite file."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    updated_files: dict[str, Path] = {}
    for file_path in sorted(data_path.glob("pbp_*.parquet")):
        match = re.match(r"^pbp_(\d{4})\.parquet$", file_path.name)
        if not match:
            continue

        season = int(match.group(1))
        df = pd.read_parquet(file_path)
        try:
            df = add_player_names(df, season=season)
        except requests.RequestException:
            # Allow offline runs when batter_name/pitcher_name already exist.
            pass
        df = add_pvc_flag(df)
        df.to_parquet(file_path, index=False)
        updated_files[file_path.stem] = file_path

    return updated_files


if __name__ == "__main__":
    scrape_and_save_statcast_pbp_years(start_year=2021, end_year=2025, data_dir="Data")
    apply_names_to_saved_pbp(data_dir="Data")
    print(pl.from_pandas(df).filter(pl.col("PVC") == 1).select(pl.col("batter_name")).unique().to_series().to_list())



