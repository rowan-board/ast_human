# plotting functions

# packages the functions use
import ast
import pandas as pd
import numpy as np
import cmdstanpy as cmd
import arviz 
import seaborn as sns
from matplotlib import pyplot as plt

# colour palette for the project
palette = {
    'human': '#2b2b2f',
    'GH': '#745B95',
    'GH (hM4Di)':'#D68AFF',
    'ISO': '#8A8C96',
    'ISO (hM3Dq)': '#B7BAC5',
    'mouse': '#993bff',

    # Charcoals & greys
    'charcoal': '#2b2b2f',
    'dark_charcoal': '#1c1c1f',
    'patient': '#3d3d44',
    'slate': '#555763',
    'cool_grey': '#727480',
    'silver': '#a6a8b3',
    'light_grey': '#d2d4dc',

    # Accents
    'plotting_1': '#00c2ff',
    'plotting_2': '#ddbdff',
}

def plot_costs(df, costs = 'shift_cost', condition = 'condition'):
    '''
    plots shift cost by group
    costs = name of shift cost column
    condition = name of condition column, set to None if using all data 
    '''
    
    sns.set_style("ticks")      # cleaner than whitegrid
    sns.set_context("paper", font_scale=1.3)

    conditions = df[condition].unique()
    plt.figure(figsize=(len(conditions) + 0.5,4))

    ax = sns.boxplot(df,
                    x = condition,
                    y = costs,
                    hue='condition',
                    palette=palette,
                    linewidth=2,
                    fliersize=False)
    ax = sns.stripplot(df,
                        x = condition,
                        y = costs,
                        hue='condition',
                        palette=palette,
                        linewidth=2)
    
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.ylabel('Number of Errors')
    plt.xlabel('')
    plt.xticks(rotation=45)
    #plt.title(string.capwords(agent))
    plt.tight_layout()


    #if groups_used[0] == 'human':
    #    plt.savefig('joint/plots/human/shift_cost_' + "_".join(groups_used) +'.svg',format='svg')
    #else: 
    #    plt.savefig('joint/plots/mouse/shift_cost_'+ "_".join(groups_used) +'..svg',format='svg')
    #return ax


def plot_trajectories(df, stage = 'stage', n_errors = 'n_errors_stage', condition = None):

    '''
    plots errors/n_trials over blocks
    df = dataframe with errors per stage for each subject, will plot all stages you include
    stage = column name for stage
    n_errors = column name for errors
    agent = where did this data come from? human/simulation/animal?
    condition = column name for group or condition you want to split by
    '''
    
    sns.set_style("ticks")      # cleaner than whitegrid
    sns.set_context("paper", font_scale=1.3)

    # get stage names
    stages = df[stage].unique()

    plt.figure(figsize=(4,4))

    if condition == None:
        ax = sns.pointplot(df,
                        x = stage,
                        y = n_errors,
                        errorbar='se',
                        order=stages,
                        color=palette['charcoal'],
                        linewidth=4)
        plt.ylim(0)
    else:
        ax = sns.pointplot(df,
                        x = stage,
                        y = n_errors,
                        errorbar='se',
                        order=stages,
                        hue='condition',
                        palette=palette,
                        linewidth=4)
        ax.get_legend().remove()
        plt.ylim(0)

    
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.ylabel('Number of Errors')
    plt.xlabel('')
    #plt.title(string.capwords(agent))
    plt.tight_layout()

    # save figure
    #if groups_used:
    #    plt.savefig('joint/plots/' + agent + '/trial_trajectory_' + "_".join(groups_used) +'.svg',format='svg')
    #else: 
    #    plt.savefig('joint/plots/' + agent + '/trial_trajectory.svg',format='svg')
    #return ax

def plot_switch(df, p_switch = 'p_switch', condition = 'condition'):
    '''
    plots shift cost by group
    costs = name of shift cost column
    condition = name of condition column, set to None if using all data 
    '''
    
    sns.set_style("ticks")      # cleaner than whitegrid
    sns.set_context("paper", font_scale=1.3)

    conditions = df[condition].unique()
    plt.figure(figsize=(len(conditions) + 0.5,4))

    ax = sns.boxplot(df,
                    x = condition,
                    y = p_switch,
                    hue='condition',
                    palette=palette,
                    linewidth=2,
                    fliersize=False)
    ax = sns.stripplot(df,
                        x = condition,
                        y = p_switch,
                        hue='condition',
                        palette=palette,
                        linewidth=2)
    
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.ylabel('p(switch after first win)')
    plt.xlabel('')
    plt.xticks(rotation=45)
    #plt.title(string.capwords(agent))
    plt.tight_layout()


    #if groups_used[0] == 'human':
    #    plt.savefig('joint/plots/human/shift_cost_' + "_".join(groups_used) +'.svg',format='svg')
    #else: 
    #    plt.savefig('joint/plots/mouse/shift_cost_'+ "_".join(groups_used) +'..svg',format='svg')
    #return ax