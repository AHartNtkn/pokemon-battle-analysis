import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from app import app

# Loat models
from joblib import load
ouModel = load('assets/ouModel.joblib')
vgcModel = load('assets/vgcModel.joblib')

# Get data for each pokemon
pokeData = pd.read_csv("datasets/smogon.csv", index_col="name")

# Used to swap team 1 and team 2
def switchTeams(x, k):
    newx = x.copy()
    
    newx[k//2:k] = x[0:k//2] # Team 1 Pokes
    newx[0:k//2] = x[k//2:k] # Team 2 Pokes

    newx[k+19:k+38] = x[k:k+19] # Team 1 Types
    newx[k:k+19] = x[k+19:k+38] # Team 2 Types

    newx[k+44:k+50] = x[k+38:k+44] # Team 1 mean
    newx[k+38:k+44] = x[k+44:k+50] # Team 2 mean

    newx[k+56:k+62] = x[k+50:k+56] # Team 1 std
    newx[k+50:k+56] = x[k+56:k+62] # Team 2 std

    newx[k+80:k+98] = x[k+62:k+80] # Team 1 typedef
    newx[k+62:k+80] = x[k+80:k+98] # Team 2 typedef

    newx[k+99] = x[k+98] # Team 1 mean types
    newx[k+98] = x[k+99] # Team 2 mean types

    newx[k+101] = x[k+100] # Team 1 immunities
    newx[k+100] = x[k+101] # Team 2 immunities

    newx[k+110:k+118] = x[k+102:k+110] # Team 1 imunity
    newx[k+102:k+110] = x[k+110:k+118] # Team 2 imunity

    newx[k+119] = x[k+118] # Team 1 superweaknesses
    newx[k+118] = x[k+119] # Team 2 superweaknesses

    newx[k+126:k+132] = x[k+120:k+126] # Team 1 Max
    newx[k+120:k+126] = x[k+126:k+132] # Team 2 Max

    newx[k+138:k+144] = x[k+132:k+138] # Team 1 Min
    newx[k+132:k+138] = x[k+138:k+144] # Team 2 Min
    
    return newx

# Used to swap team 1 and team 2 within a dataframe
def switchTeamsPd(df, k):
    df = df.copy()
    for i in range(len(df)):
        df.iloc[i] = switchTeams(df.iloc[i], k)
    
    return df

# Combine two lists of probabilities into a mean of probabilities
def combineProbs(p1, p2):
    pr = p1.copy()
    for i in range(len(p1)):
        pr[i] = [(p1[i][0]+p2[i][1])/2, (p1[i][1]+p2[i][0])/2]
    return pr

# Make prediction on an individual datapoint.
def makeIndPred(setup, k, model):
    df = pd.DataFrame([setup])
    dfS = switchTeamsPd(df, k)
    predProbs = combineProbs(model.predict_proba(df), model.predict_proba(dfS))
    pred = [ "team_1" if x[0] > .5 else "team_2" for x in predProbs ]
    return (pred[0], predProbs[0,0], predProbs[0,1])

# Calculate the stats for a row based on a team
def teamToStatsRow(team1, team2):
    pokeDataT = pokeData.T
    
    types = ["None", "Normal", "Fighting", 
         "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel", 
         "Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon",
          "Dark", "Fairy"]
    team1TypesP = np.array(list(map(lambda x: pokeDataT[x],team1)))[:,8:10].flatten()
    team2TypesP = np.array(list(map(lambda x: pokeDataT[x],team2)))[:,8:10].flatten()
    team1Types = [list(team1TypesP).count(t) for t in types]
    team2Types = [list(team2TypesP).count(t) for t in types]
    
    team1Means = list(np.array(list(map(lambda x: pokeDataT[x],team1)))[:,0:6].mean(axis=0))
    team2Means = list(np.array(list(map(lambda x: pokeDataT[x],team2)))[:,0:6].mean(axis=0))

    team1Std = list(map(lambda x: x.std(ddof=1), np.array(list(map(lambda x: pokeDataT[x],team1)))[:,0:6].T))
    team2Std = list(map(lambda x: x.std(ddof=1), np.array(list(map(lambda x: pokeDataT[x],team2)))[:,0:6].T))

    team1DefMeans = list(np.array(list(map(lambda x: pokeDataT[x],team1)))[:,14:].mean(axis=0))
    team2DefMeans = list(np.array(list(map(lambda x: pokeDataT[x],team2)))[:,14:].mean(axis=0))

    team1DefMean = [np.array(list(map(lambda x: pokeDataT[x],team1)))[:,14:].mean()]
    team2DefMean = [np.array(list(map(lambda x: pokeDataT[x],team2)))[:,14:].mean()]

    team1Immunities = [list(np.array(list(map(lambda x: pokeDataT[x],team1)))[:,14:].flatten()).count(0)]
    team2Immunities = [list(np.array(list(map(lambda x: pokeDataT[x],team2)))[:,14:].flatten()).count(0)]

    immTypes = [0, 1, 3, 4, 7, 12, 13, 15]
    team1Immunity = [ list(np.array(list(map(lambda x: pokeDataT[x],team1)))[:,14:].T[x]).count(0) for x in immTypes]
    team2Immunity = [ list(np.array(list(map(lambda x: pokeDataT[x],team2)))[:,14:].T[x]).count(0) for x in immTypes]

    team1SuperWeak = [list(np.array(list(map(lambda x: pokeDataT[x],team1)))[:,14:].flatten()).count(16)]
    team2SuperWeak = [list(np.array(list(map(lambda x: pokeDataT[x],team2)))[:,14:].flatten()).count(16)]

    team1Max = list(np.array(list(map(lambda x: pokeDataT[x],team1)))[:,0:6].max(axis=0))
    team2Max = list(np.array(list(map(lambda x: pokeDataT[x],team2)))[:,0:6].max(axis=0))

    team1Min = list(np.array(list(map(lambda x: pokeDataT[x],team1)))[:,0:6].min(axis=0))
    team2Min = list(np.array(list(map(lambda x: pokeDataT[x],team2)))[:,0:6].min(axis=0))
    
    row = team1Types + team2Types + team1Means + team2Means + team1Std + team2Std + team1DefMeans + team2DefMeans + team1DefMean + team2DefMean + team1Immunities + team2Immunities + team1Immunity + team2Immunity + team1SuperWeak + team2SuperWeak + team1Max + team2Max + team1Min + team2Min
    
    return row

# Convert an explicit team into a row compatible with the model.
def fullTeamRow(team1, team2, pokeList, index):
    return pd.Series([ x in team1 for x in pokeList ] + [ x in team2 for x in pokeList ] + teamToStatsRow(team1, team2),
                     index=index)

# The number of pokemon apearing in each dataset, times 2 (for each team)
vgcSunk = 788
ouk = 1108
# Get the indices for each dataset
ouIndex = pd.read_csv("datasets/ouIndex.csv").columns
vgcIndex = pd.read_csv("datasets/vgcSunIndex.csv").columns
# Get the list of pokemon for each dataset
vgcPokes = vgcIndex[:vgcSunk//2].str[:-7]
ouPokes = ouIndex[:ouk//2].str[:-7]

# Make individual prediction from vgc model
def vgcIndPred(team1, team2):
    return makeIndPred(fullTeamRow(team1, team2, vgcPokes, vgcIndex), vgcSunk, vgcModel)

# Make individual prediction from ou model
def ouIndPred(team1, team2):
    return makeIndPred(fullTeamRow(team1, team2, ouPokes, ouIndex), ouk, ouModel)

column1 = dbc.Col(
    [

        dcc.Markdown(
            """
        
            ## Predictions


            """
        ),

        dcc.Markdown(
            """
        
            ## Interactions go here


            """
        ),

        dcc.Markdown(
            """
        
            Description goes here.


            """
        ),
    ]
)


layout = dbc.Row([column1])