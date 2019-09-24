import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Data and Model

            The initial dataset used for training contains several thousand pokemon battles pulled from Pokémon Showdown. It only conatins the Pokémon on each team, along with the winner.

            ```
            813235259
            Excadrill,Tyranitar,Heatran,Magnezone,Rhyperior,Gallade,
            Spinda,Latios,Heatran,Serperior,Magearna,Victini,
            2
            813233608
            Blacephalon,Kartana,Chansey,Latios,Ferrothorn,Tapu Fini,
            Celesteela,Zygarde,Tapu Bulu,Tyranitar,Toxapex,Tornadus-Therian,
            [...]
            ```

            This data is then proccessed into a more usable form. Most of the data wrangling and feature engineering was done in Mathematica. Starting, the train and test data are imported, proccessed, and combined;

            ```mathematica
            dataOUTrain = Import["train_ou.txt"];
            dataOUTest = Import["test_ou.txt"];

            dataOUTrain = 
                StringSplit[#, ","] & /@ 
                    StringJoin @@@ 
                        Partition[StringSplit[dataOUTrain, "\\n"], 4][[All, 2 ;; All]];
            dataOUTest = 
                StringSplit[#, ","] & /@ 
                    StringJoin @@@ 
                        Partition[StringSplit[dataOUTest, "\\n"], 4][[All, 2 ;; All]];

            dataOU = Join[dataOUTrain, dataOUTest];
            ```

            We can take data from Smogon and turn it into an association (the Mathematica equivalent of a dictionary).

            ```mathematica
            pokeData = Import["smogon.csv"];
            statass = 
                Association[
                    Thread@Rule[pokeData[[2 ;; All, 1]], 
                        pokeData[[2 ;; All, 2 ;; All]]]];
            ```

            The names in the data need to be compatible with the Smogon data.

            ```mathematica
            dataOU = dataOU /. {"Basculin-Blue-Striped" -> "Basculin",  "Florges-Blue" -> "Florges", [...]
            ```

            Arguably, the most important aspect of data science is feature engineering. With such a dearth of features a priori, this is especially important here. By pulling data from Smogon, a bunch of metrics can automatically be calculated, such as means, standard deviation, minimum, and maximum stats, type counts, type defenses, immunities, and x4 weaknesses. Without these new features, we'd be lucky to reach 55% accuracy with our model.


            ```mathematica
            wrangle[data2_] :=
                Module[{k, data},
                data = data2;
                Do[ k = data[[g]];
                    team1 = k[[1 ;; 6]];
                    team2 = k[[7 ;; 12]];
                    
                    (*Count Totals of each type*)
                    team1Types = Tally@Flatten[(statass /@ team1)[[All, 9 ;; 10]]];
                    team1Types = 
                        Association[Rule @@@ team1Types] /@ {"None", "Normal", [...]
                    [...]
                    (*Mean stats*)
                    team1Means = N@Mean[(statass /@ team1)[[All, 1 ;; 6]]];
                    [...]
                    (*Total type imunities*)
                    team1Immunities = 
                        Count[Flatten[(statass /@ team1)[[All, 15 ;; All]]], 0];
                    [...]
                  , {g, Length@data}];
                data ]
            ```

            These data can then be converted to a csv for portability.

            ```mathematica
            dataOU = Prepend[wrangle@dataOU, featureNames];
            Export["ou.csv", dataOU]
            ```

            There are also a lot of symmetries in pokemon battles. The order of a team doesn't really matter. Which pokemon is first does matter a little, but it's more helpful to ignore team order altogether. The easiest way to implement this is a on-hot encoding of the teams.
            
            ```mathematica
            (*Function for 1-hot encoding pokemon*)
            hotEncode[dat_] := 
                Join[(Function[x, MemberQ[#, x]] /@ pokeList) &@
                    dat[[1 ;; 6]], (Function[x, MemberQ[#, x]] /@ pokeList) &@
                    dat[[7 ;; 12]]] /. {False -> 0, True -> 1}

            (*Convert OU data into one which is one-hot-encoded*)
            data = Import["ou.csv"];
            pokeList = Sort@Tally[Flatten[data[[2 ;; All, 1 ;; 12]]]][[All, 1]];
            hotPokes = 
                Prepend[hotEncode /@ data[[2 ;; All]], 
                    Join[# <> "_team_1" & /@ pokeList, # <> "_team_2" & /@ pokeList]];
            remData = data[[All, 14 ;; All]];
            winData = data[[All, 13]] /. {1 -> "team_1", 2 -> "team_2"};
            newData = Join @@@ Thread[{hotPokes, remData, Transpose@{winData}}];
            Export["ou_hot.csv", newData]
            ```

            I'll now move into Python and train a model. At the start the data is imported and split into training and validation sets.

            ```python
            oudf = pd.read_csv("ou_hot.csv")
            target = "winner"
            oufeatures = oudf.columns.drop("winner")

            ouXTrain, ouXValidate, ouYTrain, ouYValidate = train_test_split(
                oudf[oufeatures], oudf[target], train_size=0.8, test_size=0.2)
            ```

            At this point, it's worth noting that we have a baseline of 50%. That is to say, since the team orders don't matter and are randomly assigned, about half the time team 1 will win and team 2 wins the other half. By guessing that team 1 wins every time, we already obtian a 50% predicition rate. Any useful model should be able to out-perform this.

            The actual model used here is fairly simple. The features are filtered using K-best using chi-squared scores for the one-hot encoded pokemon and F-scores for the various team metrics.

            ```python
            categorical_oufeatures = oufeatures[0:ouk]
            numeric_oufeatures = oufeatures[ouk:]

            # Feature Selection Pipelines
            oufeats = ColumnTransformer([
                    ('cats', SelectKBest(score_func=chi2, k=291), categorical_oufeatures), 
                    ('nums', SelectKBest(score_func=f_classif, k=90), numeric_oufeatures)])

            categorical_vgcfeatures = vgcfeatures[0:vgcSunk]
            numeric_vgcfeatures = vgcfeatures[vgcSunk:]

            vgcfeats = ColumnTransformer([
                    ('cats', SelectKBest(score_func=chi2, k=291), categorical_vgcfeatures), 
                    ('nums', SelectKBest(score_func=f_classif, k=90), numeric_vgcfeatures)])
            ```

            After that, the data is placed into a random forest model. As previously stated, each prediction is generated by running a team through the model twice, and taking the mean of the probabilities outputed by the model.

            ```python
            ouModel = Pipeline([
                ("feats", oufeats),
                ("Rand", RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features = 0.97))
                ])

            ouModel.fit(ouXTrain, ouYTrain)
            ```

            In addition to this, whether a team is on team 1 or 2 doesn't matter since both players queue moves to be performed simultaniously. This can be accounted for by running each scenario through the model twice; the second time having the two teams switched.
            
            ```python
            # Used to swap team 1 and team 2
            def switchTeams(x, k):
                newx = x.copy()
                
                newx[k//2:k] = x[0:k//2] # Team 1 Pokes
                newx[0:k//2] = x[k//2:k] # Team 2 Pokes
                [...]
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

            # For inputs, switch teams around
            ouTeSw = switchTeamsPd(ouXValidate, ouk)
            # Run both the original and swapped teams through the model, and combine them
            predProbs = combineProbs(ouModel.predict_proba(ouXValidate), ouModel.predict_proba(ouTeSw))
            # Turn probabilities into a prediction
            pred = [ "team_1" if x[0] > .5 else "team_2" for x in predProbs ]
            # prediction accuracy
            accuracy_score(pred, ouYValidate)
            ```

            ```
            > 0.6089334548769371
            ```
            
            All this brings our accuracy to around 60%, which is about as much as can be hoped for without more detailed information.

            """
        ),

    ],
)

layout = dbc.Row([column1])