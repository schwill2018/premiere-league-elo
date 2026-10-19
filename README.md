# Premier League Elo

`Premier_League_elo_v3_fixed.ipynb` is the research notebook. `premier_league_elo.py` is the standalone daily pipeline derived from its completed final run. The notebook remains self-contained and unchanged.

## Run

Use Python 3.12 or newer:

```bash
python -m pip install -r requirements.txt
python premier_league_elo.py
```

Current-season results and fixtures come directly from the [Premier League's FPL feed](https://fantasy.premierleague.com/api/fixtures/). Full-time results update ratings on every daily run; live scores are excluded. The feed's `finished_provisional` flag means the match has ended while FPL's points review is pending, and is sufficient for a result. Completed historical seasons from [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php) are reused from `data/premier_league_matches.csv`. Failed downloads stop publication. `run_metadata.json` records the fetch time and latest result date.

`python premier_league_elo.py --offline` explicitly rebuilds exports from saved data, marks the run offline, and does not record live forecasts. Use this for verification, not scheduled publishing. `--output-dir` can select a separate directory containing copies of `model_settings.json` and `elo_parameters.csv`; `--data-dir` selects the snapshot directory.

The notebook additionally needs Jupyter, IPython, and scikit-learn; the scheduled script does not.

## Model

The script preserves the fixed notebook's 1.74 MOV numerator/intercept, 0.001 coefficient, Elo scale 400, starting rating 1000, and final tuning grids. Training begins in 2003, evaluation in 2008, with five prior seasons used for tuning. `model_settings.json` records the exact configuration.

The supplied 2008-2026 parameter selections were extracted at full precision from the completed notebook. Daily runs reuse them; a new season is tuned once. A settings mismatch stops the run rather than silently using an incompatible cache. Changing model settings requires intentionally rebuilding the cache. The production goal-margin mapping retains the notebook's absolute-error fit on evaluated matches available at publication time.

## Portfolio exports

| File | Contents |
| --- | --- |
| `current_elo_rankings.csv` | Current-season teams, rank, Elo and run date |
| `weekly_projections.csv` | Current ISO calendar week's games, forecasts and results; next available week during a gap |
| `future_projections.csv` | Available upcoming pre-kickoff forecasts |
| `daily_projections.csv` | Today's upcoming forecasts |
| `historical_predictions.csv` | Predictions genuinely recorded before kickoff |
| `performance_games.csv` | Binary diagnostic records labeled historical or deployed |
| `pl_elo_history.csv` | Evaluated Elo history, including draws |
| `elo_parameters.csv` | Selected seasonal parameters |
| `run_metadata.json` | Freshness and diagnostic metrics |

Rankings refresh daily using all completed results. Weekly predictions are recorded once before kickoff and stay fixed even as rankings change. New fixtures are added when available before kickoff; later weeks are not archived early. Completed games remain visible for their calendar week with the original forecast and updated result. A game without a genuine pregame forecast has blank prediction fields; forecasts are never backfilled after results. Daily and future projection exports use the latest ratings. Kickoff times are interpreted in Europe/London; missing times are excluded from the forecast archive.

The league feed supplies the current season's schedule. Fixtures without a confirmed kickoff are excluded until scheduled. The week field is an ISO calendar week, not an official Premier League matchweek.

`p_home` and `p_away` are expected Elo result scores (win=1, draw=0.5, loss=0), **not three-way win probabilities**. The notebook's accuracy/log-loss diagnostics exclude draws. Historical backtests and deployed performance must be displayed separately, not pooled. Historical records cover seasons before the current season. Deployed records begin when this pipeline first records forecasts and persist across season boundaries. No deployed performance is claimed for earlier games.

Completed seasons since 2003 are checked for 380 unique matches, 20 teams, and 19 home and away games per team. Refreshes cannot discard previously completed games. These checks establish historical coverage, not the provider's live update latency.

## Publishing and automation

Publish this folder as its own GitHub repository. The included `.github/workflows/premier_league_elo.yml` runs daily at **08:00 UTC** (3 AM Chicago in daylight time, 2 AM in standard time), matching NFL. It tests the pipeline, refreshes data, and commits the tables and metadata to the default branch.

After pushing, use **Actions > Update Premier League Elo > Run workflow** once and confirm success. Actions must be enabled and allowed to push to the default branch. Scheduling starts only after publication; local files alone do not activate it. GitHub may delay scheduled runs.

Old helper files and the older notebook remain locally but are ignored by Git. The fixed notebook and production pipeline are intended for this repo. Portfolio edits are a separate next step.

## Checks

```bash
python -m unittest discover -s tests -v
```
