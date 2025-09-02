import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.pbp import load_clean_nfl_pbp_playresult_data

df = load_clean_nfl_pbp_playresult_data()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# Loop through each season and team
for s in df['season'].unique():
    print(f"Calculating performance properties for season {s}")
    for p in df.query(f'season == {s}')['posteam'].unique():
        if p == None:
            continue

        # Derive the blocking property
        total_offensive_plays = len(
            df.query(f'season == {s} and posteam == "{p}"')
        )
        tackled_for_loss = len(
            df.query(f'season == {s} and posteam == "{p}" and (tackled_for_loss == 1 or sack == 1 or qb_hit == 1)')
        )
        blocking = tackled_for_loss / total_offensive_plays
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "blocking"
        ] = blocking

        # Derive the rushing property
        rush_att = df.query(f'season == {s} and posteam == "{p}" and rush_attempt == 1')
        total_rush_attempts = len(rush_att)
        total_rushing_yards = rush_att['rushing_yards'].sum()
        rushing = total_rushing_yards / total_rush_attempts
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "rushing"
        ] = rushing

        # Derive the passing property
        pass_att = df.query(f'season == {s} and posteam == "{p}" and pass_attempt == 1')
        completions = df.query(f'season == {s} and posteam == "{p}" and pass_attempt == 1 and complete_pass == 1')
        total_pass_attempts = len(pass_att)
        total_pass_completions = len(completions)
        total_pass_yards = pass_att['passing_yards'].sum()
        total_pass_touchdowns = completions['pass_touchdown'].sum()
        total_pass_interceptions = pass_att['interception'].sum()
        passing = ((
            ( # a
                ((total_pass_completions / total_pass_attempts) - 0.3) * 5
            ) +
            ( # b
                ((total_pass_yards / total_pass_attempts) - 3) * 0.25
            ) +
            ( # c
                (total_pass_touchdowns / total_pass_attempts) * 20
            ) +
            ( # d
                2.375 - ((total_pass_interceptions / total_pass_attempts) * 25)
            )
        ) / 6) * 100
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "passing"
        ] = passing

        # Derive the receiving property
        total_incompletions = total_pass_attempts - total_pass_completions
        incompletions_per_attempt = total_incompletions / total_pass_attempts
        total_yards_after_catch = completions['yards_after_catch'].sum()
        yac_per_completion = total_yards_after_catch / total_pass_completions
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "incompletions_per_attempt"
        ] = incompletions_per_attempt
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "yac_per_completion"
        ] = yac_per_completion

        # Derive the scrambling property
        scramble_att = df.query(f'season == {s} and posteam == "{p}" and qb_scramble == 1')
        total_qb_scrambles = len(scramble_att)
        total_qb_scramble_rushing = scramble_att['rushing_yards'].sum()
        scrambles_per_play = total_qb_scrambles / total_offensive_plays
        yards_per_scramble = total_qb_scramble_rushing / total_qb_scrambles
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "scrambles_per_play"
        ] = scrambles_per_play
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "yards_per_scramble"
        ] = yards_per_scramble

        # Derive the offensive_turnovers property
        total_offensive_turnovers = len(
            df.query(f'season == {s} and posteam == "{p}" and (fumble == 1 or interception == 1)')
        )
        offensive_turnovers = total_offensive_turnovers / total_offensive_plays
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "offensive_turnovers"
        ] = offensive_turnovers

        # Derive the offensive_penalties property
        total_offensive_penalties = len(
            df.query(f'season == {s} and posteam == "{p}" and penalty == 1 and penalty_team == "{p}"')
        )
        offensive_penalties = total_offensive_penalties / total_offensive_plays
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "offensive_penalties"
        ] = offensive_penalties

        # Derive the blitzing property
        total_defensive_plays = len(
            df.query(f'season == {s} and defteam == "{p}"')
        )
        made_tackle_for_loss = len(
            df.query(f'season == {s} and defteam == "{p}" and (tackled_for_loss == 1 or sack == 1 or qb_hit == 1)')
        )
        blitzing = made_tackle_for_loss / total_defensive_plays
        df.loc[
            (df['season'] == s) & (df['defteam'] == p),
            "blitzing"
        ] = blitzing

        # Derive the rush defense property
        rush_att_against = df.query(f'season == {s} and defteam == "{p}" and rush_attempt == 1')
        total_rush_attempts_against = len(rush_att_against)
        total_rushing_yards_against = rush_att['rushing_yards'].sum()
        rush_defense = total_rushing_yards_against / total_rush_attempts_against
        df.loc[
            (df['season'] == s) & (df['defteam'] == p),
            "rush_defense"
        ] = rush_defense

        # Derive the pass defense property
        pass_att_against = df.query(f'season == {s} and defteam == "{p}" and pass_attempt == 1')
        completions_against = df.query(f'season == {s} and defteam == "{p}" and pass_attempt == 1 and complete_pass == 1')
        total_pass_attempts_against = len(pass_att_against)
        total_pass_completions_against = len(completions_against)
        total_pass_yards_against = pass_att_against['passing_yards'].sum()
        total_pass_touchdowns_against = completions_against['pass_touchdown'].sum()
        total_interceptions_for = pass_att_against['interception'].sum()
        pass_defense = ((
            ( # a
                ((total_pass_completions_against / total_pass_attempts_against) - 0.3) * 5
            ) +
            ( # b
                ((total_pass_yards_against / total_pass_attempts_against) - 3) * 0.25
            ) +
            ( # c
                (total_pass_touchdowns_against / total_pass_attempts_against) * 20
            ) +
            ( # d
                2.375 - ((total_interceptions_for / total_pass_attempts_against) * 25)
            )
        ) / 6) * 100
        df.loc[
            (df['season'] == s) & (df['defteam'] == p),
            "pass_defense"
        ] = pass_defense

        # Derive the coverage property
        total_incompletions_against = total_pass_attempts_against - total_pass_completions_against
        incompletions_per_attempt_against = total_incompletions_against / total_pass_attempts_against
        total_yards_after_catch_against = completions_against['yards_after_catch'].sum()
        yac_per_completion_against = total_yards_after_catch_against / total_pass_completions_against
        df.loc[
            (df['season'] == s) & (df['defteam'] == p),
            "incompletions_per_attempt_against"
        ] = incompletions_per_attempt_against
        df.loc[
            (df['season'] == s) & (df['defteam'] == p),
            "yac_per_completion_against"
        ] = yac_per_completion_against

        # Derive the defensive_turnovers property
        total_defensive_turnovers = len(
            df.query(f'season == {s} and defteam == "{p}" and (fumble == 1 or interception == 1)')
        )
        defensive_turnovers = total_defensive_turnovers / total_defensive_plays
        df.loc[
            (df['season'] == s) & (df['defteam'] == p),
            "defensive_turnovers"
        ] = defensive_turnovers

        # Derive the defensive_penalties property
        total_defensive_penalties = len(
            df.query(f'season == {s} and defteam == "{p}" and penalty == 1 and penalty_team == "{p}"')
        )
        defensive_penalties = total_defensive_penalties / total_defensive_plays
        df.loc[
            (df['season'] == s) & (df['defteam'] == p),
            "defensive_penalties"
        ] = defensive_penalties

# Normalize the blocking property
min_blocking = df['blocking'].min()
max_blocking = df['blocking'].max()
df['norm_blocking'] = 1 - ((df['blocking'] - min_blocking) / (max_blocking - min_blocking))

# Normalize the rushing property
min_rushing = df['rushing'].min()
max_rushing = df['rushing'].max()
df['norm_rushing'] = (df['rushing'] - min_rushing) / (max_rushing - min_rushing)

# Normalize the passing property
min_passing = df['passing'].min()
max_passing = df['passing'].max()
df['norm_passing'] = (df['passing'] - min_passing) / (max_passing - min_passing)

# Normalize the receiving properties and derive normalized receiving
min_incompletions_per_attempt = df['incompletions_per_attempt'].min()
max_incompletions_per_attempt = df['incompletions_per_attempt'].max()
df['norm_incompletions_per_attempt'] = 1 - ((df['incompletions_per_attempt'] - min_incompletions_per_attempt) / (max_incompletions_per_attempt - min_incompletions_per_attempt))
min_yac_per_completion = df['yac_per_completion'].min()
max_yac_per_completion = df['yac_per_completion'].max()
df['norm_yac_per_completion'] = (df['yac_per_completion'] - min_yac_per_completion) / (max_yac_per_completion - min_yac_per_completion)
df['receiving'] = (df['norm_yac_per_completion'] + df['norm_incompletions_per_attempt']) / 2
min_receiving = df['receiving'].min()
max_receiving = df['receiving'].max()
df['norm_receiving'] = (df['receiving'] - min_receiving) / (max_receiving - min_receiving)

# Normalize the scrambling properties and derive normalized scrambling
min_scrambles_per_play = df['scrambles_per_play'].min()
max_scrambles_per_play = df['scrambles_per_play'].max()
df['norm_scrambles_per_play'] = (df['scrambles_per_play'] - min_scrambles_per_play) / (max_scrambles_per_play - min_scrambles_per_play)
min_yards_per_scramble = df['yards_per_scramble'].min()
max_yards_per_scramble = df['yards_per_scramble'].max()
df['norm_yards_per_scramble'] = (df['yards_per_scramble'] - min_yards_per_scramble) / (max_yards_per_scramble - min_yards_per_scramble)
df['scrambling'] = (df['norm_scrambles_per_play'] + df['norm_yards_per_scramble']) / 2
min_scrambling = df['scrambling'].min()
max_scrambling = df['scrambling'].max()
df['norm_scrambling'] = (df['scrambling'] - min_scrambling) / (max_scrambling - min_scrambling)

# Normalize the offensive turnovers property
min_offensive_turnovers = df['offensive_turnovers'].min()
max_offensive_turnovers = df['offensive_turnovers'].max()
df['norm_offensive_turnovers'] = 1 - ((df['offensive_turnovers'] - min_offensive_turnovers) / (max_offensive_turnovers - min_offensive_turnovers))

# Normalize the offensive penalties property
min_offensive_penalties = df['offensive_penalties'].min()
max_offensive_penalties = df['offensive_penalties'].max()
df['norm_offensive_penalties'] = 1 - ((df['offensive_penalties'] - min_offensive_penalties) / (max_offensive_penalties - min_offensive_penalties))

# Normalize the blitzing property
min_blitzing = df['blitzing'].min()
max_blitzing = df['blitzing'].max()
df['norm_blitzing'] = (df['blitzing'] - min_blitzing) / (max_blitzing - min_blitzing)

# Normalize the rush_defense property
min_rush_defense = df['rush_defense'].min()
max_rush_defense = df['rush_defense'].max()
df['norm_rush_defense'] = 1 - ((df['rush_defense'] - min_rush_defense) / (max_rush_defense - min_rush_defense))

# Normalize the pass_defense property
min_pass_defense = df['pass_defense'].min()
max_pass_defense = df['pass_defense'].max()
df['norm_pass_defense'] = 1 - ((df['pass_defense'] - min_pass_defense) / (max_pass_defense - min_pass_defense))

# Normalize the coverage properties and derive normalized coverage
min_incompletions_per_attempt_against = df['incompletions_per_attempt_against'].min()
max_incompletions_per_attempt_against = df['incompletions_per_attempt_against'].max()
df['norm_incompletions_per_attempt_against'] = (df['incompletions_per_attempt_against'] - min_incompletions_per_attempt_against) / (max_incompletions_per_attempt_against - min_incompletions_per_attempt_against)
min_yac_per_completion_against = df['yac_per_completion_against'].min()
max_yac_per_completion_against = df['yac_per_completion_against'].max()
df['norm_yac_per_completion_against'] = 1 - ((df['yac_per_completion_against'] - min_yac_per_completion_against) / (max_yac_per_completion_against - min_yac_per_completion_against))
df['coverage'] = (df['norm_yac_per_completion_against'] + df['norm_incompletions_per_attempt_against']) / 2
min_coverage = df['coverage'].min()
max_coverage = df['coverage'].max()
df['norm_coverage'] = (df['coverage'] - min_coverage) / (max_coverage - min_coverage)

# Normalize the defensive turnovers property
min_defensive_turnovers = df['defensive_turnovers'].min()
max_defensive_turnovers = df['defensive_turnovers'].max()
df['norm_defensive_turnovers'] = (df['defensive_turnovers'] - min_defensive_turnovers) / (max_defensive_turnovers - min_defensive_turnovers)

# Normalize the defensive penalties property
min_defensive_penalties = df['defensive_penalties'].min()
max_defensive_penalties = df['defensive_penalties'].max()
df['norm_defensive_penalties'] = 1 - ((df['defensive_penalties'] - min_defensive_penalties) / (max_defensive_penalties - min_defensive_penalties))

# Loop through each season and team again
grouped_off = df.groupby(['season', 'posteam'])
grouped_def = df.groupby(['season', 'defteam'])
for s in df['season'].unique():
    print(f"Calculating unit overall properties for season {s}")
    for p in df.query(f'season == {s}')['posteam'].unique():
        if p == None:
            continue

        # Calculate the team's offensive overall rating
        norm_blocking = grouped_off.get_group((s, p))['norm_blocking'].mean()
        norm_rushing = grouped_off.get_group((s, p))['norm_rushing'].mean()
        norm_passing = grouped_off.get_group((s, p))['norm_passing'].mean()
        norm_receiving = grouped_off.get_group((s, p))['norm_receiving'].mean()
        norm_offensive_turnovers = grouped_off.get_group((s, p))['norm_offensive_turnovers'].mean()
        norm_offensive_penalties = grouped_off.get_group((s, p))['norm_offensive_penalties'].mean()
        offense_overall = np.average(
            [
                norm_blocking,
                norm_rushing,
                norm_passing,
                norm_receiving,
                norm_offensive_turnovers,
                norm_offensive_penalties
            ]
        )
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "offense_overall"
        ] = offense_overall

        # Calculate the team's defensive overall rating
        norm_blitzing = grouped_def.get_group((s, p))['norm_blitzing'].mean()
        norm_rush_defense = grouped_def.get_group((s, p))['norm_rush_defense'].mean()
        norm_pass_defense = grouped_def.get_group((s, p))['norm_pass_defense'].mean()
        norm_coverage = grouped_def.get_group((s, p))['norm_coverage'].mean()
        norm_defensive_turnovers = grouped_def.get_group((s, p))['norm_defensive_turnovers'].mean()
        norm_defensive_penalties = grouped_def.get_group((s, p))['norm_defensive_penalties'].mean()
        defense_overall = np.average(
            [
                norm_blitzing,
                norm_rush_defense,
                norm_pass_defense,
                norm_coverage,
                norm_defensive_turnovers,
                norm_defensive_penalties
            ]
        )
        df.loc[
            (df['season'] == s) & (df['defteam'] == p),
            "defense_overall"
        ] = defense_overall

# Normalize the offensive and defensive overall properties
min_offense_overall = df['offense_overall'].min()
max_offense_overall = df['offense_overall'].max()
df['norm_offense_overall'] = (df['offense_overall'] - min_offense_overall) / (max_offense_overall - min_offense_overall)
min_defense_overall = df['defense_overall'].min()
max_defense_overall = df['defense_overall'].max()
df['norm_defense_overall'] = (df['defense_overall'] - min_defense_overall) / (max_defense_overall - min_defense_overall)

# Loop through each season and team a third time
grouped_off = df.groupby(['season', 'posteam'])
grouped_def = df.groupby(['season', 'defteam'])
for s in df['season'].unique():
    print(f"Calculating overall properties for season {s}")
    for p in df.query(f'season == {s}')['posteam'].unique():
        if p == None:
            continue

        # Get the team's offense and defense overall
        offense_overall = grouped_off.get_group((s, p))['norm_offense_overall'].mean()
        defense_overall = grouped_def.get_group((s, p))['norm_defense_overall'].mean()

        # Calculate the team's overall rating
        overall = np.average(
            [
                offense_overall,
                defense_overall
            ]
        )
        df.loc[
            (df['season'] == s) & (df['posteam'] == p),
            "overall"
        ] = overall

# Normalize the team overall properties
min_overall = df['overall'].min()
max_overall = df['overall'].max()
df['norm_overall'] = (df['overall'] - min_overall) / (max_overall - min_overall)

# Visualize the derived properties
off_plot_df = df[
    [
        'season',
        'posteam',
        'norm_overall',
        'norm_offense_overall',
        'norm_blocking',
        'norm_rushing',
        'norm_passing',
        'norm_receiving',
        'norm_scrambling',
        'norm_offensive_turnovers',
        'norm_offensive_penalties'
    ]
].drop_duplicates()
def_plot_df = df[
    [
        'season',
        'defteam',
        'norm_defense_overall',
        'norm_blitzing',
        'norm_rush_defense',
        'norm_pass_defense',
        'norm_coverage',
        'norm_defensive_turnovers',
        'norm_defensive_penalties'
    ]
].drop_duplicates()

# Visualize the overall property
off_plot_df['norm_overall'].hist(bins=20)
plt.title("Team overall distribution")
plt.savefig('./figures/overall.png')
plt.clf()

# Visualize the offense overall property
off_plot_df['norm_offense_overall'].hist(bins=20)
plt.title("Offense overall distribution")
plt.savefig('./figures/offense_overall.png')
plt.clf()

# Visualize the defense overall property
def_plot_df['norm_defense_overall'].hist(bins=20)
plt.title("Defense overall distribution")
plt.savefig('./figures/defense_overall.png')
plt.clf()

# Visualize the blocking property
off_plot_df['norm_blocking'].hist(bins=20)
plt.title("Blocking distribution")
plt.savefig('./figures/blocking.png')
plt.clf()

# Visualize the rushing property
off_plot_df['norm_rushing'].hist(bins=20)
plt.title("Rushing distribution")
plt.savefig('./figures/rushing.png')
plt.clf()

# Visualize the passing property
off_plot_df['norm_passing'].hist(bins=20)
plt.title("Passing distribution")
plt.savefig('./figures/passing.png')
plt.clf()

# Visualize the receiving property
off_plot_df['norm_receiving'].hist(bins=20)
plt.title("Receiving distribution")
plt.savefig('./figures/receiving.png')
plt.clf()

# Visualize the scrambling property
off_plot_df['norm_scrambling'].hist(bins=20)
plt.title("Scrambling distribution")
plt.savefig('./figures/scrambling.png')
plt.clf()

# Visualize the offensive turnovers property
off_plot_df['norm_offensive_turnovers'].hist(bins=20)
plt.title("Offensive turnovers distribution")
plt.savefig('./figures/off_turnovers.png')
plt.clf()

# Visualize the offensive penalties property
off_plot_df['norm_offensive_penalties'].hist(bins=20)
plt.title("Offensive penalties distribution")
plt.savefig('./figures/off_penalties.png')
plt.clf()

# Visualize the blitzing property
def_plot_df['norm_blitzing'].hist(bins=20)
plt.title("Blitzing distribution")
plt.savefig('./figures/blitzing.png')
plt.clf()

# Visualize the rush defense property
def_plot_df['norm_rush_defense'].hist(bins=20)
plt.title("Rush defense distribution")
plt.savefig('./figures/rush_defense.png')
plt.clf()

# Visualize the pass defense property
def_plot_df['norm_pass_defense'].hist(bins=20)
plt.title("Pass defense distribution")
plt.savefig('./figures/pass_defense.png')
plt.clf()

# Visualize the coverage property
def_plot_df['norm_coverage'].hist(bins=20)
plt.title("Coverage distribution")
plt.savefig('./figures/coverage.png')
plt.clf()

# Visualize the defensive turnovers property
def_plot_df['norm_defensive_turnovers'].hist(bins=20)
plt.title("Defensive turnovers distribution")
plt.savefig('./figures/def_turnovers.png')
plt.clf()

# Visualize the defensive penalties property
def_plot_df['norm_defensive_penalties'].hist(bins=20)
plt.title("Defensive penalties distribution")
plt.savefig('./figures/def_penalties.png')
plt.clf()
