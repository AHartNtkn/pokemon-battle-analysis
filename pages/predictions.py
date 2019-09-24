import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from app import app

# Load models
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
    return predProbs[0]

# Calculate the stats for a row based on a team
def teamToStatsRow(team1, team2):
    pokeDataT = pokeData.T
    
    types = ["None", "Normal", "Fighting",  "Flying", "Poison", "Ground",
             "Rock", "Bug", "Ghost", "Steel",  "Fire", "Water", "Grass",
             "Electric", "Psychic", "Ice", "Dragon", "Dark", "Fairy"]
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

# The number of pokemon appearing in each dataset, times 2 (1 for each team)
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

# Lists for generating dropdown menus
vgcDrop = list(map(lambda x: {'label': x, 'value': x}, vgcPokes))
ouDrop = list(map(lambda x: {'label': x, 'value': x}, ouPokes))

@app.callback(
    [Output('team-1-percent-vgc', 'children'),
     Output('team-2-percent-vgc', 'children'),
     Output('team-1-percent-ou', 'children'),
     Output('team-2-percent-ou', 'children'),
     Output('vgc_icon_team1_1', 'src'), Output('vgc_icon_team1_2', 'src'), Output('vgc_icon_team1_3', 'src'),
     Output('vgc_icon_team1_4', 'src'), Output('vgc_icon_team1_5', 'src'), Output('vgc_icon_team1_6', 'src'),
     Output('vgc_icon_team2_1', 'src'), Output('vgc_icon_team2_2', 'src'), Output('vgc_icon_team2_3', 'src'),
     Output('vgc_icon_team2_4', 'src'), Output('vgc_icon_team2_5', 'src'), Output('vgc_icon_team2_6', 'src'),
     Output('ou_icon_team1_1', 'src'), Output('ou_icon_team1_2', 'src'), Output('ou_icon_team1_3', 'src'),
     Output('ou_icon_team1_4', 'src'), Output('ou_icon_team1_5', 'src'), Output('ou_icon_team1_6', 'src'),
     Output('ou_icon_team2_1', 'src'), Output('ou_icon_team2_2', 'src'), Output('ou_icon_team2_3', 'src'),
     Output('ou_icon_team2_4', 'src'), Output('ou_icon_team2_5', 'src'), Output('ou_icon_team2_6', 'src')],
    [Input('vgc_team1_1', 'value'), Input('vgc_team1_2', 'value'), Input('vgc_team1_3', 'value'),
     Input('vgc_team1_4', 'value'), Input('vgc_team1_5', 'value'), Input('vgc_team1_6', 'value'),
     Input('vgc_team2_1', 'value'), Input('vgc_team2_2', 'value'), Input('vgc_team2_3', 'value'),
     Input('vgc_team2_4', 'value'), Input('vgc_team2_5', 'value'), Input('vgc_team2_6', 'value'),
     Input('ou_team1_1', 'value'), Input('ou_team1_2', 'value'), Input('ou_team1_3', 'value'),
     Input('ou_team1_4', 'value'), Input('ou_team1_5', 'value'), Input('ou_team1_6', 'value'),
     Input('ou_team2_1', 'value'), Input('ou_team2_2', 'value'), Input('ou_team2_3', 'value'),
     Input('ou_team2_4', 'value'), Input('ou_team2_5', 'value'), Input('ou_team2_6', 'value')
    ])
def multi_output(vgc_team1_1, vgc_team1_2, vgc_team1_3, vgc_team1_4, vgc_team1_5, vgc_team1_6,
                 vgc_team2_1, vgc_team2_2, vgc_team2_3, vgc_team2_4, vgc_team2_5, vgc_team2_6,
                 ou_team1_1, ou_team1_2, ou_team1_3, ou_team1_4, ou_team1_5, ou_team1_6,
                 ou_team2_1, ou_team2_2, ou_team2_3, ou_team2_4, ou_team2_5, ou_team2_6):
    predVGC = vgcIndPred([vgc_team1_1, vgc_team1_2, vgc_team1_3, vgc_team1_4, vgc_team1_5, vgc_team1_6],
                         [vgc_team2_1, vgc_team2_2, vgc_team2_3, vgc_team2_4, vgc_team2_5, vgc_team2_6])
    predOU = ouIndPred([ou_team1_1, ou_team1_2, ou_team1_3, ou_team1_4, ou_team1_5, ou_team1_6],
                      [ou_team2_1, ou_team2_2, ou_team2_3, ou_team2_4, ou_team2_5, ou_team2_6])

    return (f'Team 1 Chance at winning: {100*predVGC[0]:.1f}%', f'Team 2 Chance at winning: {100*predVGC[1]:.1f}%',
            f'Team 1 Chance at winning: {100*predOU[0]:.1f}%', f'Team 2 Chance at winning: {100*predOU[1]:.1f}%',
            "assets/icons/" + vgc_team1_1 + ".png", "assets/icons/" + vgc_team1_2 + ".png", "assets/icons/" + vgc_team1_3 + ".png", "assets/icons/" + vgc_team1_4 + ".png", "assets/icons/" + vgc_team1_5 + ".png", "assets/icons/" + vgc_team1_6 + ".png",
            "assets/icons/" + vgc_team2_1 + ".png", "assets/icons/" + vgc_team2_2 + ".png", "assets/icons/" + vgc_team2_3 + ".png", "assets/icons/" + vgc_team2_4 + ".png", "assets/icons/" + vgc_team2_5 + ".png", "assets/icons/" + vgc_team2_6 + ".png",
            "assets/icons/" + ou_team1_1 + ".png", "assets/icons/" + ou_team1_2 + ".png", "assets/icons/" + ou_team1_3 + ".png", "assets/icons/" + ou_team1_4 + ".png", "assets/icons/" + ou_team1_5 + ".png", "assets/icons/" + ou_team1_6 + ".png",
            "assets/icons/" + ou_team2_1 + ".png", "assets/icons/" + ou_team2_2 + ".png", "assets/icons/" + ou_team2_3 + ".png", "assets/icons/" + ou_team2_4 + ".png", "assets/icons/" + ou_team2_5 + ".png", "assets/icons/" + ou_team2_6 + ".png")

vgcCol1 = dbc.Col([
    dbc.Row([
            html.Img(id='vgc_icon_team1_1',width="75"),
            html.Img(id='vgc_icon_team1_2',width="75"),
            html.Img(id='vgc_icon_team1_3',width="75"),
            html.Img(id='vgc_icon_team1_4',width="75"),
            html.Img(id='vgc_icon_team1_5',width="75"),
            html.Img(id='vgc_icon_team1_6',width="75"),
            ]),
    dbc.Row([
            dcc.Dropdown(
                id='vgc_team1_1', 
                options = vgcDrop, 
                value = vgcPokes[0], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team1_2', 
                options = vgcDrop, 
                value = vgcPokes[1], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team1_3', 
                options = vgcDrop, 
                value = vgcPokes[2], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team1_4', 
                options = vgcDrop, 
                value = vgcPokes[3], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team1_5', 
                options = vgcDrop, 
                value = vgcPokes[4], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team1_6', 
                options = vgcDrop, 
                value = vgcPokes[5], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
        ]),
])

vgcCol2 = dbc.Col([
    dbc.Row([
            html.Img(id='vgc_icon_team2_1',width="75"),
            html.Img(id='vgc_icon_team2_2',width="75"),
            html.Img(id='vgc_icon_team2_3',width="75"),
            html.Img(id='vgc_icon_team2_4',width="75"),
            html.Img(id='vgc_icon_team2_5',width="75"),
            html.Img(id='vgc_icon_team2_6',width="75"),
            ]),

    dbc.Row([
            dcc.Dropdown(
                id='vgc_team2_1', 
                options = vgcDrop, 
                value = vgcPokes[0], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team2_2', 
                options = vgcDrop, 
                value = vgcPokes[1], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team2_3', 
                options = vgcDrop, 
                value = vgcPokes[2], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team2_4', 
                options = vgcDrop, 
                value = vgcPokes[3], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team2_5', 
                options = vgcDrop, 
                value = vgcPokes[4], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='vgc_team2_6', 
                options = vgcDrop, 
                value = vgcPokes[5], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
        ]),
])

ouCol1 = dbc.Col([
        dbc.Row([
            html.Img(id='ou_icon_team1_1',width="75"),
            html.Img(id='ou_icon_team1_2',width="75"),
            html.Img(id='ou_icon_team1_3',width="75"),
            html.Img(id='ou_icon_team1_4',width="75"),
            html.Img(id='ou_icon_team1_5',width="75"),
            html.Img(id='ou_icon_team1_6',width="75"),
            ]),

        dbc.Row([
            dcc.Dropdown(
                id='ou_team1_1', 
                options = ouDrop, 
                value = ouPokes[0], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            
            dcc.Dropdown(
                id='ou_team1_2', 
                options = ouDrop, 
                value = ouPokes[1], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            
            dcc.Dropdown(
                id='ou_team1_3', 
                options = ouDrop, 
                value = ouPokes[2], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            
            dcc.Dropdown(
                id='ou_team1_4', 
                options = ouDrop, 
                value = ouPokes[3], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            
            dcc.Dropdown(
                id='ou_team1_5', 
                options = ouDrop, 
                value = ouPokes[4], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            
            dcc.Dropdown(
                id='ou_team1_6', 
                options = ouDrop, 
                value = ouPokes[5], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
        ]),
])

ouCol2 = dbc.Col([
    dbc.Row([
            html.Img(id='ou_icon_team2_1',width="75"),
            html.Img(id='ou_icon_team2_2',width="75"),
            html.Img(id='ou_icon_team2_3',width="75"),
            html.Img(id='ou_icon_team2_4',width="75"),
            html.Img(id='ou_icon_team2_5',width="75"),
            html.Img(id='ou_icon_team2_6',width="75"),
            ]),

        dbc.Row([
            dcc.Dropdown(
                id='ou_team2_1', 
                options = ouDrop, 
                value = ouPokes[0], 
                className='mb-5', 
                style={  'width': '75px', "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='ou_team2_2', 
                options = ouDrop, 
                value = ouPokes[1], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='ou_team2_3', 
                options = ouDrop, 
                value = ouPokes[2], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='ou_team2_4', 
                options = ouDrop, 
                value = ouPokes[3], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='ou_team2_5', 
                options = ouDrop, 
                value = ouPokes[4], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
            dcc.Dropdown(
                id='ou_team2_6', 
                options = ouDrop, 
                value = ouPokes[5], 
                className='mb-5', 
                style={ 'width': '75px',  "font-size": "10px"}
            ),
        ])
])

vgcCol = dbc.Col([
    dcc.Markdown(
            """
        
            # VGC Ultra Sun and Moon Predictor

            """
        ),

    dbc.Row([
        dbc.Col([
            html.H2("Team 1")
        ]),

        dbc.Col([
            html.H2("Team 2")
        ]),
    ]),

    dbc.Row([
        vgcCol1,
        vgcCol2
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id='team-1-percent-vgc', className='lead')
        ]),

        dbc.Col([
            html.Div(id='team-2-percent-vgc', className='lead')
        ]),
    ]),
])

ouCol = dbc.Col([
    dcc.Markdown(
            """
        
            # OU Predictor

            """
        ),
    dbc.Row([
        dbc.Col([
            html.H2("Team 1")
        ]),

        dbc.Col([
            html.H2("Team 2")
        ]),
    ]),

    
    dbc.Row([
        ouCol1,
        ouCol2
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='team-1-percent-ou', className='lead')
        ]),

        dbc.Col([
            html.Div(id='team-2-percent-ou', className='lead')
        ]),
    ]),
])
  
desCol = dbc.Col([
        dcc.Markdown(
            """
        
            Description goes here.


            """
        ),
    ])

layout = dbc.Col([vgcCol, html.Hr(), ouCol, html.Hr(), desCol])