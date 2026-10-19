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
        with patch.object(pl, 'download', side_effect=TimeoutError('offline')):
            with self.assertRaises(TimeoutError):
                pl.load_games(pl.ROOT / 'data', pd.Timestamp('2026-10-19', tz='Europe/London'))

    def test_settings_mismatch_fails(self):
        with patch.object(Path, 'read_text', return_value='{}'):
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                pl.fit_model(self.games, pl.ROOT)

    def test_spread_fit_excludes_current_season_results(self):
        history = pd.DataFrame(dict(season=[2025, 2025, 2026],
                                    home_score=[2., 0., 100.], away_score=[0., 2., 0.],
                                    p_home=[.75, .25, .75]))
        slope = pl.fit_spread_slope(history, 2026)
        self.assertAlmostEqual(slope, 2 / np.log(3), places=5)
        history.loc[2, ['home_score', 'away_score', 'p_home']] = [0., 500., .01]
        self.assertEqual(slope, pl.fit_spread_slope(history, 2026))

    def test_seasonal_spread_is_saved_and_reused_without_another_fit(self):
        cache = pd.read_csv(pl.ROOT / 'elo_parameters.csv', float_precision='round_trip')
        cache['spread_slope'] = np.nan
        cache.loc[cache.eval_year.eq(2025), 'spread_slope'] = .731
        with patch.object(pl.pd, 'read_csv', return_value=cache), \
                patch.object(pl, 'fit_spread_slope', wraps=pl.fit_spread_slope) as fit:
            first = pl.fit_model(self.games, pl.ROOT)
        fit.assert_called_once()
        saved = first['K_by_year']
        self.assertEqual(saved.loc[saved.eval_year.eq(2025), 'spread_slope'].iloc[0], .731)
        slope = saved.loc[saved.eval_year.eq(2026), 'spread_slope'].iloc[0]
        self.assertAlmostEqual(slope, .7283451363667883, places=12)
        changed = self.games.copy()
        current = changed.index[changed.season.eq(2026) & changed.home_score.notna()][0]
        changed.loc[current, ['home_score', 'away_score']] = [10., 0.]
        with patch.object(pl.pd, 'read_csv', return_value=saved), \
                patch.object(pl, 'minimize_scalar', side_effect=AssertionError('Unexpected refit')):
            second = pl.fit_model(changed, pl.ROOT)
            empty = pd.DataFrame(columns=['game_id', 'gameday', 'gametime', 'p_home', 'prediction_timestamp'])
            outputs, metrics = pl.make_outputs(changed, second, empty,
                pd.Timestamp('2026-10-19T08:00Z'), record_predictions=False)
        self.assertEqual(metrics['production_margin_slope'], slope)
        self.assertEqual(outputs['elo_parameters.csv'].loc[
            outputs['elo_parameters.csv'].eval_year.eq(2026), 'spread_slope'].iloc[0], slope)

    def test_weekly_archive_is_frozen_even_when_ratings_rank_or_kickoff_change(self):
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
        pd.testing.assert_frame_equal(first, third)
        rescheduled = fixture.assign(gameday='2026-10-25', gametime='16:30', home_rank=3)
        fourth = pl.update_archive(third, rescheduled, pd.Timestamp('2026-10-23T10:00Z'))
        pd.testing.assert_frame_equal(first, fourth)
        after = pl.update_archive(third, fixture.assign(home_elo=900., p_home=.3), pd.Timestamp('2026-10-24T16:00Z'))
        pd.testing.assert_frame_equal(third, after)

    def test_official_feed_uses_full_time_results_and_ignores_live_scores(self):
        teams = [dict(id=i + 1, name=name) for i, name in enumerate([
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea',
            'Coventry City', 'Crystal Palace', 'Everton', 'Fulham', 'Hull City', 'Ipswich Town',
            'Leeds', 'Liverpool', 'Man City', 'Man Utd', 'Newcastle', "Nott'm Forest", 'Spurs', 'Sunderland'])]
        fixtures = [
            dict(event=7, kickoff_time='2026-10-18T13:00:00Z', team_h=5, team_a=8, team_h_score=5,
                 team_a_score=0, finished=False, finished_provisional=True),
            dict(event=7, kickoff_time='2026-10-19T19:00:00Z', team_h=19, team_a=7, team_h_score=1,
                 team_a_score=0, finished=False, finished_provisional=False),
            dict(event=8, kickoff_time=None, team_h=11, team_a=12, team_h_score=None,
                 team_a_score=None, finished=False, finished_provisional=False)]
        with patch.object(pl, 'download', side_effect=[json.dumps(fixtures), json.dumps(dict(teams=teams))]):
            games = pl.load_current_season(2026)
        self.assertEqual(len(games), 2)
        completed = games.dropna(subset=['home_score', 'away_score'])
        self.assertEqual(list(completed.home_team), ['Brighton'])
        self.assertEqual(completed.iloc[0].home_score, 5)
        live = games[games.home_team.eq('Tottenham')].iloc[0]
        self.assertTrue(pd.isna(live.home_score))
        self.assertEqual(live.away_team, 'Coventry')
        self.assertEqual(live.gametime, '20:00')
        self.assertTrue(games.week.eq(7).all())
        with patch.object(pl, 'download', side_effect=[json.dumps(fixtures), json.dumps(dict(teams=teams))]):
            with self.assertRaisesRegex(ValueError, 'not for season'):
                pl.load_current_season(2025)

    def test_new_results_change_rankings_without_rewriting_weekly_forecasts(self):
        games = self.games[self.games.gameday.le('2026-10-12') & self.games.home_score.notna()].copy()
        # A league matchday spans Saturday through Monday, across a calendar boundary.
        new_games = pl.normalize_matches(pd.DataFrame([
            dict(Div='E0', Matchweek=7, Date='17/10/2026', Time='12:30', HomeTeam='Everton', AwayTeam='Chelsea'),
            dict(Div='E0', Matchweek=7, Date='18/10/2026', Time='14:00', HomeTeam='Brighton', AwayTeam='Crystal Palace'),
            dict(Div='E0', Matchweek=7, Date='19/10/2026', Time='20:00', HomeTeam='Tottenham', AwayTeam='Coventry'),
            dict(Div='E0', Matchweek=8, Date='23/10/2026', Time='20:00', HomeTeam='Ipswich', AwayTeam="Nott'm Forest")]), 2026)
        games = pd.concat([games, new_games], ignore_index=True)
        empty = pd.DataFrame(columns=['game_id', 'gameday', 'gametime', 'p_home', 'prediction_timestamp'])
        first, _ = pl.make_outputs(games, pl.fit_model(games, pl.ROOT), empty,
                                    pd.Timestamp('2026-10-16T08:00Z'))
        archive = first['historical_predictions.csv']
        # The following week's game must not freeze early with this week's ratings.
        self.assertEqual(set(archive.game_id), {'2026_Everton_Chelsea', '2026_Brighton_Crystal Palace',
                                              '2026_Tottenham_Coventry'})
        games.loc[games.game_id.eq('2026_Everton_Chelsea'), ['home_score', 'away_score']] = [0., 2.]
        games.loc[games.game_id.eq('2026_Brighton_Crystal Palace'), ['home_score', 'away_score']] = [5., 0.]
        # Exercise the live loader as well as the calculation: past results must survive refresh.
        with patch.object(pl.pd, 'read_csv', return_value=games[games.gameday.le('2026-10-12')]), \
                patch.object(pl, 'load_current_season', return_value=games[games.season.eq(2026)]):
            refreshed, metadata = pl.load_games(pl.ROOT / 'data', pd.Timestamp('2026-10-18T20:00Z'))
        self.assertEqual(metadata['latest_result_date'], '2026-10-18')
        second, _ = pl.make_outputs(refreshed, pl.fit_model(refreshed, pl.ROOT), archive,
                                     pd.Timestamp('2026-10-18T20:00Z'))
        pd.testing.assert_frame_equal(archive, second['historical_predictions.csv'])
        old = first['current_elo_rankings.csv'].set_index('team')
        new = second['current_elo_rankings.csv'].set_index('team')
        self.assertGreater(new.loc['Brighton', 'elo'], old.loc['Brighton', 'elo'])
        self.assertLess(new.loc['Crystal Palace', 'elo'], old.loc['Crystal Palace', 'elo'])
        self.assertGreater(new.loc['Chelsea', 'elo'], old.loc['Chelsea', 'elo'])
        weekly = second['weekly_projections.csv'].set_index('game_id')
        self.assertEqual(weekly.loc['2026_Brighton_Crystal Palace', 'result'], 'HOME')
        self.assertEqual(weekly.loc['2026_Brighton_Crystal Palace', 'p_home'],
                         archive.set_index('game_id').loc['2026_Brighton_Crystal Palace', 'p_home'])
        # Monday's unfinished game keeps matchday 7 active; matchday 8 stays separate.
        third, _ = pl.make_outputs(games, pl.fit_model(games, pl.ROOT), archive,
                                    pd.Timestamp('2026-10-19T20:00Z'))
        self.assertTrue(third['weekly_projections.csv'].week.eq(7).all())
        self.assertEqual(len(third['weekly_projections.csv']), 3)
        self.assertTrue(third['weekly_projections.csv'].p_home.notna().all())
        games.loc[games.game_id.eq('2026_Tottenham_Coventry'), ['home_score', 'away_score']] = [2., 0.]
        fourth, _ = pl.make_outputs(games, pl.fit_model(games, pl.ROOT), archive,
                                     pd.Timestamp('2026-10-20T08:00Z'))
        self.assertTrue(fourth['weekly_projections.csv'].week.eq(8).all())
        self.assertEqual(list(fourth['weekly_projections.csv'].game_id), ['2026_Ipswich_Nott\'m Forest'])

    def test_unknown_kickoff_is_not_archived(self):
        frame = pd.DataFrame([dict(game_id='unknown', gameday='2026-10-24', gametime='00:00', p_home=.6)])
        empty = pd.DataFrame(columns=list(frame.columns) + ['prediction_timestamp'])
        self.assertTrue(pl.update_archive(empty, frame, pd.Timestamp('2026-10-20T10:00Z')).empty)

    def test_full_forecast_to_scored_result(self):
        games = self.games[self.games.gameday.le('2026-10-12') & self.games.home_score.notna()].copy()
        fixture = pl.normalize_matches(pd.DataFrame([dict(
            Div='E0', Matchweek=7, Date='19/10/2026', Time='20:00', HomeTeam='Tottenham',
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
        self.assertTrue(weekly.week.eq(6).all())
        self.assertTrue(weekly.p_home.isna().all())
        self.assertTrue(outputs['performance_games.csv'].evaluation_type.eq('historical').all())


if __name__ == '__main__':
    unittest.main()
