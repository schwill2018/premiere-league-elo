"""Premier League Elo: notebook model plus scheduled CSV publishing.

The model functions come from Premier_League_elo_v3_fixed.ipynb.
Run python premier_league_elo.py --help for usage.
"""
import argparse
import io
import json
import time
from itertools import product
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
FPL_FIXTURES = 'https://fantasy.premierleague.com/api/fixtures/'
FPL_TEAMS = 'https://fantasy.premierleague.com/api/bootstrap-static/'
SETTINGS = {
    "model_version": 1, "mov_numerator": 1.74, "mov_coefficient": 0.001,
    "scale": 400, "start_rating": 1000,
    "first_train_year": 2003, "first_eval_year": 2008,
    "K_grid": [15, 18, 21, 24, 27, 30, 33, 36],
    "home_adv_grid": [45, 50, 55, 65, 70, 75, 80, 85, 95, 100],
    "carryover_grid": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
    "param_tune_window": 5,
}

def download(url):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 2:
                raise
            print(f"Retrying {url}: {exc}", flush=True)
            time.sleep(2 ** attempt)


def read_football_csv(url):
    payload = download(url)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("cp1252")  # Some historical files use Windows encoding.
    # Older season files contain surplus trailing commas. Selecting the
    # needed columns tolerates those empty fields without skipping matches.
    columns = {"Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    return pd.read_csv(io.StringIO(text), usecols=lambda column: column in columns)


def normalize_matches(raw, season=None):
    raw = raw.loc[raw["Div"].eq("E0")].copy()
    raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    out = pd.DataFrame(index=raw.index)
    out["gameday"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="raise")
    out["gametime"] = raw["Time"].fillna("00:00") if "Time" in raw else "00:00"
    out["season"] = (season if season is not None else
                     out["gameday"].dt.year - (out["gameday"].dt.month < 7).astype(int))
    out["home_team"] = raw["HomeTeam"].str.strip()
    out["away_team"] = raw["AwayTeam"].str.strip()
    for target, source in [("home_score", "FTHG"), ("away_score", "FTAG")]:
        out[target] = pd.to_numeric(raw[source], errors="raise") if source in raw else np.nan
    out["location"] = "Home"
    out["game_type"] = "PL"
    out["week"] = (pd.to_numeric(raw["Matchweek"], errors="raise").astype("Int64")
                   if "Matchweek" in raw else out["gameday"].dt.strftime("%G-W%V"))
    out["game_id"] = (out["season"].astype(str) + "_" + out["home_team"] + "_" + out["away_team"])
    return out.reset_index(drop=True)


def load_current_season(season):
    """Use the league's results directly; Football-Data's CSV can lag by days."""
    fixtures = json.loads(download(FPL_FIXTURES))
    teams = json.loads(download(FPL_TEAMS))['teams']
    # Keep the same team keys as the historical Football-Data ratings.
    aliases = {'Man Utd': 'Man United', 'Spurs': 'Tottenham',
               'Coventry City': 'Coventry', 'Hull City': 'Hull', 'Ipswich Town': 'Ipswich'}
    names = {team['id']: aliases.get(team['name'], team['name']) for team in teams}
    frame = pd.DataFrame(fixtures)
    if frame.empty or len(names) != 20:
        raise ValueError('Empty or invalid Premier League current-season feed')
    kickoff = pd.to_datetime(frame.kickoff_time, utc=True).dt.tz_convert('Europe/London')
    dated = kickoff.dropna()
    if dated.empty or not (dated.dt.year - (dated.dt.month < 7).astype(int)).eq(season).all():
        raise ValueError(f'Premier League feed is not for season {season}')
    # FPL marks full time provisionally before its fantasy-points review finishes.
    # A live match can already have scores, so scores alone never imply full time.
    finished = frame.finished | frame.finished_provisional
    if frame.loc[finished, ['team_h_score', 'team_a_score']].isna().any().any():
        raise ValueError('Completed Premier League match is missing a score')
    raw = pd.DataFrame({'Div': 'E0', 'Date': kickoff.dt.strftime('%d/%m/%Y'),
                        'Matchweek': frame.event,
                        'Time': kickoff.dt.strftime('%H:%M'),
                        'HomeTeam': frame.team_h.map(names), 'AwayTeam': frame.team_a.map(names),
                        'FTHG': frame.team_h_score.where(finished),
                        'FTAG': frame.team_a_score.where(finished)})
    if raw[['HomeTeam', 'AwayTeam']].isna().any().any():
        raise ValueError('Unknown team in Premier League feed')
    return normalize_matches(raw, season)


def null_coalesce(x, y):
    return y if x is None else x


def compress_exp_grid(min_val, max_val, n=10, curve=2):
    grid = np.unique(np.round(
        min_val + (max_val - min_val) *
        (np.exp(curve * np.linspace(0, 1, n)) - 1) / (np.exp(curve) - 1),2))
    return grid


def calc_mov_scale(games):
    mean_log_margin = np.log(
        (games["home_score"] - games["away_score"]).abs() + 1
    ).mean()
    return mean_log_margin


def pl_home_result(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return np.nan
    return 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5


def elo_season(games, K, ratings=None, scale=400, start_rating=1000,
               use_mov=True, mov_scale=1, use_home_adv=True, home_adv=0):

    ratings = {} if ratings is None else ratings.copy()
    n = len(games)

    if use_mov and (pd.isna(mov_scale) or mov_scale <= 0):
        mov_scale = 1
    if not use_home_adv:
        home_adv = 0

    elo_home_pre = np.zeros(n)
    elo_away_pre = np.zeros(n)
    elo_home_post = np.zeros(n)
    elo_away_post = np.zeros(n)
    p_home = np.zeros(n)

    for i, g in enumerate(games.to_dict("records")):

        hid = str(g["home_team"])
        aid = str(g["away_team"])

        Ra = ratings.get(hid, start_rating)
        Rb = ratings.get(aid, start_rating)

        elo_home_pre[i] = Ra
        elo_away_pre[i] = Rb

        game_home_adv = 0 if g["location"] == "Neutral" else home_adv

        p = 1 / (1 + 10 ** ((Rb - Ra - game_home_adv) / scale))
        p_home[i] = p

        y = pl_home_result(home_score=g["home_score"], away_score=g["away_score"])

        if pd.isna(y):
            raise ValueError(f"Invalid Premier League result coding for game_id: {g['game_id']}")

        goal_diff = abs(g["home_score"] - g["away_score"])
        if g["home_score"] > g["away_score"]:
            winner_elo_diff = (Ra + game_home_adv) - Rb
        elif g["home_score"] < g["away_score"]:
            winner_elo_diff = Rb - (Ra + game_home_adv)
        else:
            winner_elo_diff = 0

        if not use_mov:
          mov_mult = 1
        elif g["home_score"] == g["away_score"]:
          mov_mult = 1
        else:
          mov_mult = np.log(goal_diff + 1) * (1.74 / (0.001 * winner_elo_diff + 1.74)) / mov_scale

        delta = K * (y - p) * mov_mult

        ratings[hid] = Ra + delta
        ratings[aid] = Rb - delta

        elo_home_post[i] = ratings[hid]
        elo_away_post[i] = ratings[aid]

    games_out = games.copy()
    games_out["elo_home_pre"] = elo_home_pre
    games_out["elo_away_pre"] = elo_away_pre
    games_out["elo_home_post"] = elo_home_post
    games_out["elo_away_post"] = elo_away_post
    games_out["p_home"] = p_home
    games_out["p_away"] = 1 - p_home

    return {"games": games_out, "ratings": ratings}


def carry_over(ratings, carry=0.6):
    if len(ratings) == 0:
        return ratings

    mu = np.mean(list(ratings.values()))
    return {team: carry * rating + (1 - carry) * mu
            for team, rating in ratings.items()}


def logloss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def score_elo_params(all_games, seasons_tune, K, home_adv,
                     use_mov=True, mov_scale=1, use_home_adv=True,
                     scale=400, start_rating=1000, carry=0.6,
                     initial_ratings=None):

  ratings = {} if initial_ratings is None else initial_ratings.copy()
  ll_vec = []

  if not use_home_adv:
      home_adv = 0

  for s in seasons_tune:
      g_s = (all_games[all_games["season"] == s]
              .sort_values(["gameday", "game_id"])
              .reset_index(drop=True))

      ratings = carry_over(ratings, carry=carry)
      res = elo_season(games=g_s, ratings=ratings, K=K, scale=scale,
                       start_rating=start_rating, use_mov=use_mov,
                       mov_scale=mov_scale, use_home_adv=use_home_adv,
                       home_adv=home_adv)

      games_res = res["games"]
      non_ties = games_res["home_score"] != games_res["away_score"]

      p = games_res.loc[non_ties, "p_home"].to_numpy()
      y = (games_res.loc[non_ties, "home_score"] >
            games_res.loc[non_ties, "away_score"]).astype(float).to_numpy()

      ll_vec.extend(logloss(p, y))
      ratings = res["ratings"]

  return np.nanmean(ll_vec)


def tune_elo_params(all_games, seasons_tune, K_grid, home_adv_grid,
                    use_mov=True, mov_scale=1, use_home_adv=True,
                    scale=400, start_rating=1000, carry=0.6,
                    use_carry_grid=True,
                    carryover_grid=(0.35, 0.50, 0.65, 0.80),
                    initial_ratings=None):

    if not use_home_adv:
        home_adv_grid = [0]

    carry_grid = carryover_grid if use_carry_grid else [carry]
    results = []

    for K, home_adv, carry_val in product(K_grid, home_adv_grid, carry_grid):
        ll = score_elo_params(
            all_games=all_games, seasons_tune=seasons_tune,
            K=K, home_adv=home_adv,
            use_mov=use_mov, mov_scale=mov_scale,
            use_home_adv=use_home_adv, scale=scale,
            start_rating=start_rating, carry=carry_val,
            initial_ratings=initial_ratings
        )

        results.append({"K": K, "home_adv": home_adv, "carry": carry_val,
                        "logloss": ll})

    results = pd.DataFrame(results).sort_values("logloss").reset_index(drop=True)

    best = results.iloc[0]

    return {"K_best": best["K"], "home_adv_best": best["home_adv"],
            "carry_best": best["carry"], "logloss": best["logloss"],
            "results": results}


def run_expanding_k_elo(all_games, first_train_year=2003, first_eval_year=2010,
                        K_grid=None, home_adv_grid=None, scale=400,
                        start_rating=1000, carry=0.6, use_carry_grid=True,
                        carryover_grid=(0.35, 0.50, 0.65, 0.80),
                        param_tune_window=None, use_mov=True, use_home_adv=True,
                        projection_season=None, saved_parameters=None):

    if K_grid is None:
        K_grid = compress_exp_grid(1, 60, n=10, curve=-1)
    if home_adv_grid is None:
        home_adv_grid = compress_exp_grid(1, 60, n=10, curve=1)
    if not use_home_adv:
        home_adv_grid = [0]

    K_grid = np.atleast_1d(K_grid)
    home_adv_grid = np.atleast_1d(home_adv_grid)
    fixed_param_mode = len(K_grid) == 1 and len(home_adv_grid) == 1 and not use_carry_grid

    seasons = sorted(all_games["season"].unique())
    eval_years = [s for s in seasons if s >= first_eval_year]

    # Include the current schedule season even if it has no completed games yet.
    if projection_season is not None and projection_season not in eval_years:
        eval_years = sorted(set(eval_years + [projection_season]))

    all_out = []
    param_out = []
    ratings = {}
    ratings_before_season = {}

    for yr in eval_years:
        train_years = [s for s in seasons if first_train_year <= s < yr]
        if len(train_years) == 0:
            raise ValueError(f"No training seasons available before eval year: {yr}")

        param_tune_years = (
            train_years if param_tune_window is None
            else train_years[-param_tune_window:]
        )

        train_games = all_games[all_games["season"].isin(train_years)]
        mov_scale = calc_mov_scale(train_games) if use_mov else np.nan

        if saved_parameters is not None and yr in saved_parameters.index:
            cached = saved_parameters.loc[yr]
            K_best = cached['K_best']
            home_adv_best = cached['home_adv_best']
            carry_best = cached['carry_best']
            mov_scale = cached['mov_scale']
            best_logloss = cached['logloss']
        elif fixed_param_mode:
            K_best = K_grid[0]
            home_adv_best = home_adv_grid[0] if use_home_adv else 0
            carry_best = carry
            best_logloss = np.nan
        else:
            param_fit = tune_elo_params(
                all_games=all_games, seasons_tune=param_tune_years,
                K_grid=K_grid, home_adv_grid=home_adv_grid,
                use_mov=use_mov, mov_scale=mov_scale,
                use_home_adv=use_home_adv, scale=scale,
                start_rating=start_rating, carry=carry,
                use_carry_grid=use_carry_grid,
                carryover_grid=carryover_grid,
                initial_ratings=ratings_before_season.get(param_tune_years[0], {})
            )

            K_best = param_fit["K_best"]
            home_adv_best = param_fit["home_adv_best"] if use_home_adv else 0
            carry_best = param_fit["carry_best"] if use_carry_grid else carry
            best_logloss = param_fit["logloss"]

        # Bootstrap training history once; later seasons keep the prior final ratings.
        if yr == eval_years[0]:
            for s in train_years:
                season_games = (all_games[all_games["season"] == s]
                                .sort_values(["gameday", "gametime", "game_id"])
                                .reset_index(drop=True))

                ratings_before_season[s] = ratings.copy()
                ratings = carry_over(ratings, carry=carry_best)
                res = elo_season(
                    games=season_games, ratings=ratings, K=K_best,
                    scale=scale, start_rating=start_rating,
                    use_mov=use_mov, mov_scale=mov_scale,
                    use_home_adv=use_home_adv, home_adv=home_adv_best
                )

                ratings = res["ratings"]

        ratings_before_season[yr] = ratings.copy()
        ratings = carry_over(ratings, carry=carry_best)

        eval_games = (all_games[all_games["season"] == yr]
                      .sort_values(["gameday", "gametime", "game_id"])
                      .reset_index(drop=True))

        res_eval = elo_season(
            games=eval_games, ratings=ratings, K=K_best,
            scale=scale, start_rating=start_rating,
            use_mov=use_mov, mov_scale=mov_scale,
            use_home_adv=use_home_adv, home_adv=home_adv_best
        )

        ratings = res_eval["ratings"]
        games_out = res_eval["games"].copy()
        games_out["K_used"] = K_best
        games_out["home_adv_used"] = np.where(games_out["location"] == "Neutral", 0, home_adv_best)
        games_out["carry_used"] = carry_best
        games_out["mov_scale_used"] = mov_scale
        games_out["use_mov"] = use_mov
        games_out["use_home_adv"] = use_home_adv
        games_out["train_start"] = min(train_years)
        games_out["train_end"] = max(train_years)
        games_out["param_tune_start"] = min(param_tune_years)
        games_out["param_tune_end"] = max(param_tune_years)
        games_out["param_tune_n"] = len(param_tune_years)
        games_out["param_tune_window"] = param_tune_window

        all_out.append(games_out)

        param_out.append({
            "eval_year": yr,
            "train_start": min(train_years),
            "train_end": max(train_years),
            "param_tune_start": min(param_tune_years),
            "param_tune_end": max(param_tune_years),
            "param_tune_n": len(param_tune_years),
            "param_tune_window": param_tune_window,
            "K_best": K_best,
            "home_adv_best": home_adv_best,
            "carry_best": carry_best,
            "mov_scale": mov_scale,
            "use_mov": use_mov,
            "use_home_adv": use_home_adv,
            "use_carry_grid": use_carry_grid,
            "logloss": best_logloss
        })

    return {
        "games": pd.concat(all_out, ignore_index=True),
        "K_by_year": pd.DataFrame(param_out),
        "ratings": res_eval["ratings"],
        "current_season": eval_years[-1]
    }


def validate_games(games, current_season):
    if games.empty or games['game_id'].duplicated().any():
        raise ValueError('Empty schedule or duplicate game IDs')
    if (games.home_score.isna() != games.away_score.isna()).any():
        raise ValueError('A match has only one score')
    completed = games.dropna(subset=['home_score', 'away_score'])
    scores = completed[['home_score', 'away_score']].to_numpy()
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores % 1 != 0).any():
        raise ValueError('Invalid goals in downloaded results')
    for season in range(2003, current_season):
        frame = completed[completed.season.eq(season)]
        teams = set(frame.home_team) | set(frame.away_team)
        if (len(frame) != 380 or len(teams) != 20
                or not frame.home_team.value_counts().eq(19).all()
                or not frame.away_team.value_counts().eq(19).all()
                or frame[['home_team', 'away_team']].duplicated().any()):
            raise ValueError(f'Incomplete season {season}: expected 380 matches, 20 teams, 19 home/away each')
    return completed


def load_games(data_dir, now, offline=False):
    """Live runs must fetch successfully; offline runs are explicit and never archived."""
    snapshot = data_dir / 'premier_league_matches.csv'
    season = now.year if now.month >= 7 else now.year - 1
    previous = pd.read_csv(snapshot, parse_dates=['gameday']) if snapshot.exists() else None
    if offline:
        if previous is None:
            raise FileNotFoundError(snapshot)
        validate_games(previous, season)
        metadata_path = data_dir / 'snapshot.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        return previous, metadata
    frames = []
    for year in range(2003, season + 1):
        if year == season:
            print(f'Fetching official Premier League season {year}/{year + 1}', flush=True)
            frames.append(load_current_season(year))
            continue
        old = previous[previous.season.eq(year)] if previous is not None else pd.DataFrame()
        if year < season and len(old.dropna(subset=['home_score', 'away_score'])) == 380:
            frames.append(old.dropna(subset=['home_score', 'away_score']).copy())
            continue
        code = f'{year % 100:02d}{(year + 1) % 100:02d}'
        print(f'Fetching season {year}/{year + 1}', flush=True)
        frame = normalize_matches(read_football_csv(f'https://www.football-data.co.uk/mmz4281/{code}/E0.csv'), year)
        if frame.empty:
            raise ValueError(f'Empty season feed: {year}')
        frames.append(frame)
    for frame in frames:
        if frame.game_id.duplicated().any():
            raise ValueError('Duplicate match within a source feed')
    games = pd.concat(frames, ignore_index=True)
    games['completed'] = games.home_score.notna() & games.away_score.notna()
    games = (games.sort_values('completed', ascending=False)
             .drop_duplicates('game_id').sort_values(['gameday', 'gametime', 'game_id']).reset_index(drop=True))
    completed = validate_games(games, season)
    if previous is not None:
        old_ids = set(previous.dropna(subset=['home_score', 'away_score']).game_id)
        if not old_ids.issubset(set(completed.game_id)):
            raise ValueError('Refresh lost previously completed matches; existing exports kept')
    metadata = {
        'downloaded_at': now.isoformat(),
        'source': FPL_FIXTURES,
        'historical_source': 'https://www.football-data.co.uk/englandm.php',
        'fixtures_source': FPL_FIXTURES,
        'latest_result_date': str(completed.gameday.max().date()),
        'completed_matches': len(completed),
        'season_counts': {str(k): int(v) for k, v in completed.groupby('season').size().items()},
    }
    return games, metadata


def fit_model(games, output_dir):
    settings_path = output_dir / 'model_settings.json'
    cache_path = output_dir / 'elo_parameters.csv'
    if not settings_path.exists() or json.loads(settings_path.read_text()) != SETTINGS:
        raise ValueError('Model settings differ from the saved parameter cache. Review/rebuild the cache before publishing.')
    params = pd.read_csv(cache_path, float_precision='round_trip')
    if params.eval_year.duplicated().any():
        raise ValueError('Duplicate cached parameter years')
    if 'train_end' in params and not params.train_end.lt(params.eval_year).all():
        raise ValueError('Cached parameters include their evaluation season in training')
    if not np.isfinite(params[['K_best', 'home_adv_best', 'carry_best', 'mov_scale']]).all().all():
        raise ValueError('Invalid cached parameters')
    completed = games.dropna(subset=['home_score', 'away_score']).copy()
    model = run_expanding_k_elo(
        completed, first_train_year=2003, first_eval_year=2008,
        K_grid=SETTINGS['K_grid'], home_adv_grid=SETTINGS['home_adv_grid'],
        carryover_grid=SETTINGS['carryover_grid'], param_tune_window=5,
        projection_season=int(games.season.max()), saved_parameters=params.set_index('eval_year'))
    # Store the spread conversion with the existing seasonal Elo parameter cache.
    saved = params.set_index('eval_year').get('spread_slope', pd.Series(dtype=float))
    if not np.isfinite(saved.dropna()).all() or not saved.dropna().between(0, 20).all():
        raise ValueError('Invalid cached spread multiplier')
    table = model['K_by_year']
    table['spread_slope'] = table.eval_year.map(saved)
    current = table.eval_year.eq(model['current_season'])
    if table.loc[current, 'spread_slope'].isna().any():
        table.loc[current, 'spread_slope'] = fit_spread_slope(model['games'], model['current_season'])
    return model


def fit_spread_slope(history, season):
    """Fit the existing absolute-error conversion using earlier seasons only."""
    train = history[history.season.lt(season)]
    if train.empty:
        raise ValueError('No prior-season games for the spread fit')
    margin = train.home_score - train.away_score
    p = train.p_home.clip(1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p))
    result = minimize_scalar(lambda s: np.mean(np.abs(margin - s * logits)),
                             bounds=(0, 20), method='bounded')
    if not result.success or not np.isfinite(result.x):
        raise ValueError('Spread fit did not converge')
    return float(result.x)


def kickoff_utc(frame):
    local = pd.to_datetime(frame.gameday.astype(str).str[:10] + ' ' + frame.gametime.astype(str), errors='coerce')
    # Unknown midnight fallback must not be treated as a verified kickoff.
    local = local.mask(frame.gametime.isin(['00:00', 'nan', '']))
    return local.dt.tz_localize('Europe/London', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')


def update_archive(existing, upcoming, now):
    """Record each weekly forecast once, before kickoff; never rewrite it."""
    fresh = upcoming[kickoff_utc(upcoming).gt(now) & ~upcoming.game_id.isin(existing.game_id)].copy()
    fresh['prediction_timestamp'] = now.isoformat()
    return (pd.concat([existing, fresh], ignore_index=True)
            .drop_duplicates('game_id', keep='last').reset_index(drop=True))


def make_outputs(games, model, existing, now, record_predictions=True):
    season = model['current_season']
    params = model['K_by_year'].iloc[-1]
    history = model['games'].copy()
    teams = sorted(set(games.loc[games.season.eq(season), 'home_team']) |
                   set(games.loc[games.season.eq(season), 'away_team']))
    ranks = pd.DataFrame({'team': teams, 'elo': [model['ratings'].get(t, 1000) for t in teams]})
    ranks = ranks.sort_values(['elo', 'team'], ascending=[False, True]).reset_index(drop=True)
    ranks.insert(0, 'rank', np.arange(1, len(ranks) + 1))
    ranks['as_of_date'] = str(now.date())
    slope = float(params.spread_slope)
    future = games[games.season.eq(season) & games.home_score.isna() & games.away_score.isna()].copy()
    future = future[kickoff_utc(future).gt(now)].copy()
    ratings = ranks.set_index('team').elo
    rank_map = ranks.set_index('team')['rank']
    for side in ['home', 'away']:
        future[f'{side}_elo'] = future[f'{side}_team'].map(ratings)
        future[f'{side}_rank'] = future[f'{side}_team'].map(rank_map)
    ha = np.where(future.location.eq('Neutral'), 0, params.home_adv_best)
    future['p_home'] = 1 / (1 + 10 ** ((future.away_elo - future.home_elo - ha) / 400))
    future['p_away'] = 1 - future.p_home
    p = future.p_home.clip(1e-6, 1 - 1e-6)
    future['elo_home_margin'] = slope * np.log(p / (1 - p))
    future['model_home_spread'] = -np.round(future.elo_home_margin * 2) / 2
    future['as_of_date'] = str(now.date())
    cols = ['as_of_date', 'season', 'week', 'game_id', 'gameday', 'gametime', 'away_team', 'home_team',
            'away_rank', 'home_rank', 'away_elo', 'home_elo', 'p_away', 'p_home', 'elo_home_margin', 'model_home_spread']
    future = future[cols].sort_values(['gameday', 'gametime', 'game_id']).reset_index(drop=True)
    # Like NFL, keep the first unfinished league matchday, including its completed games.
    current = games[games.season.eq(season)].copy()
    current['week'] = pd.to_numeric(current.week, errors='raise').astype('Int64')
    future['week'] = pd.to_numeric(future.week, errors='raise').astype('Int64')
    unfinished = current.loc[current.home_score.isna(), 'week'].dropna()
    week = unfinished.min() if not unfinished.empty else current.week.max()
    archive = (update_archive(existing, future[future.week.eq(week)], now)
               if record_predictions else existing.copy())
    weekly = current[current.week.eq(week)].copy()
    forecast_cols = ['game_id', 'as_of_date', 'away_rank', 'home_rank', 'away_elo', 'home_elo',
                     'p_away', 'p_home', 'elo_home_margin', 'model_home_spread']
    weekly = weekly.merge(archive.reindex(columns=forecast_cols), on='game_id', how='left')
    weekly['result'] = np.where(weekly.home_score.isna(), 'TBD', np.where(
        weekly.home_score.eq(weekly.away_score), 'DRAW', np.where(weekly.home_score.gt(weekly.away_score), 'HOME', 'AWAY')))
    base_cols = ['season', 'week', 'game_id', 'gameday', 'away_team', 'home_team', 'p_home', 'home_score', 'away_score']
    historical = history[history.season.lt(season)][base_cols].copy()
    historical['evaluation_type'] = 'historical'
    results = games.dropna(subset=['home_score', 'away_score'])
    deployed = results.drop(columns=['p_home'], errors='ignore').merge(
        archive[['game_id', 'p_home', 'prediction_timestamp']], on='game_id', how='inner')
    deployed = deployed[pd.to_datetime(deployed.prediction_timestamp, utc=True).lt(kickoff_utc(deployed))]
    # Retain deployed records across season boundaries; separate categories prevent double counting.
    deployed = deployed[base_cols].copy()
    deployed['evaluation_type'] = 'deployed'
    performance = pd.concat([historical, deployed], ignore_index=True)
    performance = performance[performance.home_score.ne(performance.away_score)].copy()
    performance['actual_home_win'] = performance.home_score.gt(performance.away_score).astype(int)
    if performance.duplicated(['evaluation_type', 'game_id']).any():
        raise ValueError('Duplicate performance records')
    if not performance.p_home.between(0, 1).all():
        raise ValueError('Invalid performance predictions')
    decisive = history[history.home_score.ne(history.away_score)]
    metrics = {
        'notebook_diagnostic_accuracy': float(((decisive.p_home >= .5) == (decisive.home_score > decisive.away_score)).mean()),
        'notebook_diagnostic_logloss': float(logloss(decisive.p_home, (decisive.home_score > decisive.away_score).astype(float)).mean()),
        'notebook_diagnostic_games': len(decisive),
        'draws_excluded_from_binary_diagnostic': int(history.home_score.eq(history.away_score).sum()),
        'production_margin_slope': float(slope),
        'deployed_scored_games': int(performance.evaluation_type.eq('deployed').sum()),
    }
    return {
        'current_elo_rankings.csv': ranks, 'future_projections.csv': future,
        'daily_projections.csv': future[pd.to_datetime(future.gameday).dt.date.eq(now.date())],
        'weekly_projections.csv': weekly, 'historical_predictions.csv': archive,
        'performance_games.csv': performance, 'pl_elo_history.csv': history,
        'elo_parameters.csv': model['K_by_year'],
    }, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use saved data for verification; do not record live forecasts')
    parser.add_argument('--output-dir', type=Path, default=ROOT, help='Directory containing settings/parameter cache and exports')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    now = pd.Timestamp.now(tz='Europe/London')
    games, source_metadata = load_games(args.data_dir, now, args.offline)
    model = fit_model(games, args.output_dir)
    archive_path = args.output_dir / 'historical_predictions.csv'
    archive = (pd.read_csv(archive_path, float_precision='round_trip') if archive_path.exists() else pd.DataFrame(
        columns=['game_id', 'gameday', 'gametime', 'p_home', 'prediction_timestamp']))
    outputs, metrics = make_outputs(games, model, archive, now, record_predictions=not args.offline)
    # All calculations/validation finish before any published output is replaced.
    for name, frame in outputs.items():
        path = args.output_dir / name
        staged = path.with_suffix('.csv.tmp')
        frame.to_csv(staged, index=False)
        staged.replace(path)
    if not args.offline:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        games.to_csv(args.data_dir / 'premier_league_matches.csv', index=False)
        (args.data_dir / 'snapshot.json').write_text(json.dumps(source_metadata, indent=2) + '\n')
    metadata = dict(source_metadata, generated_at=now.isoformat(), offline=args.offline,
                    current_season=int(model['current_season']), metrics=metrics,
                    prediction_meaning='Expected result score: win=1, draw=0.5, loss=0; not three-way win probability')
    (args.output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
