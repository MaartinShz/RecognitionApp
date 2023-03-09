import plotly.graph_objects as go

def plot_emotion_wheel(emotion_scores):
    # Trier les scores par ordre décroissant pour obtenir les émotions les plus probables
    sorted_scores = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

    # Définir les couleurs pour chaque quadrant de la Roue des émotions
    colors = {'Positive': '#00a8cc', 'Neutral': '#6c757d', 'Negative': '#ff6b6b'}

    # Définir les émotions pour chaque quadrant de la Roue des émotions
    emotions = {'Positive': ['happy', 'surprise'], 'Neutral': ['neutral'], 'Negative': ['angry', 'sad']}

    # Récupérer l'émotion la plus probable
    most_likely_emotion = sorted_scores[0][0]

    # Déterminer le quadrant de la Roue des émotions correspondant à l'émotion la plus probable
    for quadrant, emotions_list in emotions.items():
        if most_likely_emotion in emotions_list:
            break

    # Placer l'émotion sur la Roue des émotions en fonction de son intensité
    fig = go.Figure()
    fig.add_trace(go.Pie(values=[emotion_scores[most_likely_emotion], 1-emotion_scores[most_likely_emotion]], 
                        hole=.7, 
                        marker_colors=['#f2f2f2', colors[quadrant]],
                        textinfo='none',
                        direction='clockwise',
                        rotation=0))
    fig.update_traces(hoverinfo='none', textfont_size=18)
    fig.update_layout(
        annotations=[
            dict(
                text=most_likely_emotion.capitalize(),
                font=dict(size=32, color=colors[quadrant]),
                showarrow=False,
                x=0.5,
                y=0.5
            )
        ],
        width=300,
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white'
    )
    return fig
