import ast
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import premier_league_elo as pl


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = pd.read_csv(pl.ROOT / 'data/premier_league_matches.csv', parse_dates=['gameday'])

    def test_model_matches_executed_notebook(self):
        notebook = json.loads((pl.ROOT / 'Premier_League_elo_v3_fixed.ipynb').read_text(encoding='utf-8'))
        namespace = dict(np=np, pd=pd, product=pl.product)
        for i in [7, 11, 15, 20, 25]:
            source = ''.join(notebook['cells'][i]['source'])
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    exec(ast.get_source_segment(source, node), namespace)
        # Run the independent notebook path with its recorded choices instead of
        # spending minutes repeating the grid search during every scheduled job.
        table = next(o['data']['application/vnd.microsoft.datawrangler.viewer.v0+json']
                     for o in notebook['cells'][47]['outputs']
                     if 'application/vnd.microsoft.datawrangler.viewer.v0+json' in o.get('data', {}))
        selected = pd.DataFrame(table['rows'], columns=[c['name'] for c in table['columns']]).astype(float)
        selected = selected.rename(columns={'tuning_logloss': 'logloss'}).set_index('eval_year')
        namespace['tune_elo_params'] = lambda **kw: selected.loc[max(kw['seasons_tune']) + 1].to_dict()
        completed = self.games.dropna(subset=['home_score', 'away_score'])
        # Fixed checkpoint excludes later daily results, making this test stable.
        completed = completed[completed.gameday.le('2026-10-12') & completed.season.le(2026)]
        notebook_result = namespace['run_expanding_k_elo'](
            completed, first_eval_year=2008, param_tune_window=5)
        production_result = pl.run_expanding_k_elo(
            completed, first_eval_year=2008, param_tune_window=5, saved_parameters=selected)
        self.assertEqual(notebook_result['ratings'], production_result['ratings'])
        decisive = production_result['games'].query('home_score != away_score')
        loss = pl.logloss(decisive.p_home, (decisive.home_score > decisive.away_score).astype(float)).mean()
        self.assertAlmostEqual(loss, 0.565636565112149, places=12)

    def test_missing_historical_match_fails(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete season 2003'):
            pl.validate_games(self.games.iloc[1:], int(self.games.season.max()))

    def test_partial_score_fails(self):
        games = self.games.copy()
        games.loc[0, 'away_score'] = np.nan
        with self.assertRaisesRegex(ValueError, 'only one score'):
            pl.validate_games(games, int(games.season.max()))

    def test_live_download_failure_does_not_fallback(self):
        with patch.object(pl, 'read_football_csv', side_effect=TimeoutError('offline')):
            with self.assertRaises(TimeoutError):
                pl.load_games(pl.ROOT / 'data', pd.Timestamp('2026-10-19', tz='Europe/London'))

    def test_settings_mismatch_fails(self):
        with patch.object(Path, 'read_text', return_value='{}'):
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                pl.fit_model(self.games, pl.ROOT)

    def test_archive_stability_changes_and_kickoff_lock(self):
        fixture = pd.DataFrame([dict(game_id='match', gameday='2026-10-24', gametime='15:00',
                                     home_elo=1100., away_elo=1000., home_rank=1, away_rank=2,
                                     p_home=.6)])
        empty = pd.DataFrame(columns=list(fixture.columns) + ['prediction_timestamp'])
        first = pl.update_archive(empty, fixture, pd.Timestamp('2026-10-20T10:00Z'))
        unchanged = fixture.assign(p_home=.61)  # Margin recalibration alone must not rewrite a forecast.
        second = pl.update_archive(first, unchanged, pd.Timestamp('2026-10-21T10:00Z'))
        pd.testing.assert_frame_equal(first, second)
        changed = fixture.assign(home_elo=1120., p_home=.64)
        third = pl.update_archive(second, changed, pd.Timestamp('2026-10-22T10:00Z'))
        self.assertEqual(third.iloc[0].p_home, .64)
        after = pl.update_archive(third, fixture.assign(home_elo=900., p_home=.3), pd.Timestamp('2026-10-24T16:00Z'))
        pd.testing.assert_frame_equal(third, after)

    def test_unknown_kickoff_is_not_archived(self):
        frame = pd.DataFrame([dict(game_id='unknown', gameday='2026-10-24', gametime='00:00', p_home=.6)])
        empty = pd.DataFrame(columns=list(frame.columns) + ['prediction_timestamp'])
        self.assertTrue(pl.update_archive(empty, frame, pd.Timestamp('2026-10-20T10:00Z')).empty)

    def test_full_forecast_to_scored_result(self):
        games = self.games[self.games.gameday.le('2026-10-12') & self.games.home_score.notna()].copy()
        fixture = pl.normalize_matches(pd.DataFrame([dict(
            Div='E0', Date='19/10/2026', Time='20:00', HomeTeam='Tottenham',
            AwayTeam='Coventry', FTHG=np.nan, FTAG=np.nan)]), 2026)
        games = pd.concat([games, fixture], ignore_index=True)
        model = pl.fit_model(games, pl.ROOT)
        empty = pd.DataFrame(columns=['game_id', 'gameday', 'gametime', 'p_home', 'prediction_timestamp'])
        first, _ = pl.make_outputs(games, model, empty, pd.Timestamp('2026-10-19T08:00Z'))
        archive = first['historical_predictions.csv']
        self.assertEqual(len(archive), 1)
        original_p = archive.iloc[0].p_home
        # Publish a result and rerun the ratings after kickoff.
        games.loc[games.game_id.eq('2026_Tottenham_Coventry'), ['home_score', 'away_score']] = [2., 0.]
        next_model = pl.fit_model(games, pl.ROOT)
        second, _ = pl.make_outputs(games, next_model, archive, pd.Timestamp('2026-10-20T08:00Z'))
        self.assertEqual(second['historical_predictions.csv'].iloc[0].p_home, original_p)
        weekly = second['weekly_projections.csv']
        self.assertEqual(weekly.iloc[0].p_home, original_p)
        self.assertEqual(weekly.iloc[0].result, 'HOME')
        deployed = second['performance_games.csv'].query("evaluation_type == 'deployed'")
        self.assertEqual(len(deployed), 1)
        self.assertEqual(deployed.iloc[0].p_home, original_p)

    def test_week_keeps_completed_matches_without_hindsight_forecasts(self):
        games = self.games[self.games.gameday.le('2026-10-12')].copy()
        # Use the existing snapshot's final completed week as a controlled date.
        model = pl.fit_model(games, pl.ROOT)
        empty = pd.DataFrame(columns=['game_id', 'gameday', 'gametime', 'p_home', 'prediction_timestamp'])
        outputs, _ = pl.make_outputs(games, model, empty, pd.Timestamp('2026-10-13', tz='Europe/London'))
        weekly = outputs['weekly_projections.csv']
        self.assertTrue(weekly.week.eq('2026-W42').all())
        self.assertTrue(weekly.p_home.isna().all())
        self.assertTrue(outputs['performance_games.csv'].evaluation_type.eq('historical').all())


if __name__ == '__main__':
    unittest.main()
