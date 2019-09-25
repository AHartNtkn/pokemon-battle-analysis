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
        
            ## Insights and Limitations

            With this model, we might want to know which attributes are most important when it comes to determining a winner. We can get some insight into this by looking at the permutation importance. Here's the top twenty for the VGC Sun and Moon predictor.

            """
        ),

        html.Div([
            html.Img(src="assets/perm_importance.png")],
            style={"textAlign": "center"}),

        dcc.Markdown(
            """
        
            Most of the most important features are engineered ones, specifically stat mean and standard deviations. This is, perhaps, not too surprising, as it is the most generic way of summarizing a team's contents, in the abstract. Of less importance are the type-related features. The exact types of the Pokémon matter a lot less than how defensive a team is against another type. Among those, the exact important ones depend on the league. In OU, having good Fighting defense or Dragon immunity matters a lot more, while Ground and Fairy defense tends to matter more for VGC Sun and Moon. Good Steel defense matters in both.
            
            It's relatively rare for a single Pokémon species to have a dramatic impact on outcomes, but a few do come up. We see Tapu Fini in the list above. The most influential Pokémon tend to be legendaries; the various Tapus, Yveltal, Kyogre and Groudon, Ho-oh and Lugia along with Ultra Beasts, and a handful of others. A smaller number of non-legendary Pokémon also end of making a big impact. These include Toxapex, Garchomp, Lurantis, Incineroar, Tyranitar, Snorlax, and a handful of others. 

            """
        ),

        dcc.Markdown(
            """

            The model, in its current form, has many limitations. It was stuck training on specific leagues. This limited the pool of possibilities that it could observe. Many exotic options would not appear in the dataset at all, so its mostly useful for assessing the effectiveness of existing setups. Since all teams which appeared in the dataset were designed by a human, the model is limited in it's ability to aid in strategy discovery.

            Most limitations of the model stem from the dataset itself. Initially, it only contained the Pokémon on each team, along with the winner. From this, many team metrics can be generated, but this has its limits. The dataset doesn't contain information which is often essential for inferring the strategy underlying a teams construction. For example, it doesn't mention held items, moves, effort values, or special abilities. Each Pokémon comes in a multitude of variations which can dramatically effect the outcome of a battle.
            
            Perhaps even more import than the lack of information on individual Pokémon is the lack of information on the battlers. Having some kind of proxy for skill, such as some kind of Elo rating, would likely help improve accuracy more than any other individual metric. More speculatively, one might be able to render a short description of a team strategy into a vector using NLP, which might help improve accuracy.

            """
        ),
    ]
)

layout = dbc.Row([column1])