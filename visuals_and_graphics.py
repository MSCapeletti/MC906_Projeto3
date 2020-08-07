import plotly.graph_objects as go
import pandas as pd

def fig_real_predicted_values(real_data, predicted_data, title=None):
    fig = go.Figure()
    fig.add_trace((
        go.Scatter(
            x=real_data['Date'].apply(lambda x: x.strftime('%Y-%m-%d')), 
            y=real_data['Close'], 
            mode='lines', 
            name='Historic'
        )
    ))

    fig.add_trace((
        go.Scatter(
            x=predicted_data['Date'].apply(lambda x: x.strftime('%Y-%m-%d')), 
            y=predicted_data['Close'], 
            mode='lines', name='Predicted'
        )
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Close",
    )

    if title != None:
        fig.update_layout(
            title=title
        )   

    fig.show()


def fig_sct_open_close(real_data):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=real_data['Open'], 
            y=real_data['Close'], 
            mode='markers', 
            name='Close Values by Open Values'
        )
    )

    # Linha y=x, apenas para situar melhor os pontos do scatter acima
    fig.add_trace(
        go.Scatter(
            x=real_data['Open'].unique(), 
            y=real_data['Open'].unique(), 
            mode='lines', 
            name='Close = Open'
        )
    )

    fig.update_layout(
        xaxis_title="Open",
        yaxis_title="Close",
    )

    fig.show()