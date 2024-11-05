import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import calendar
import os

# Sample data loading
# Replace 'your_data.csv' with the path to your CSV file, or adjust accordingly if you have the DataFrame in memory
data = pd.read_csv('../datasets/FE_Final_Data.csv')

# Aggregate data by year, month, and state
data['month'] = pd.to_datetime(data['month'], format='%m').dt.month  # Assuming 'month' column is numeric (1-12)
data_avg = data.groupby(['year', 'month', 'state'], as_index=False).agg({'sales': 'mean'})

# Create a random prediction DataFrame from 2013 to 2023 with monthly data
years = np.arange(2013, 2024)
states = data['state'].unique()
predictions = []

for state in states:
    for year in years:
        for month in range(1, 13):
            predictions.append({
                'year': year,
                'month': month,
                'state': state,
                'predicted_sales': np.random.randint(4000, 6000)  # Random sales values
            })

predict_df = pd.DataFrame(predictions)

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Electricity Usage Dashboard (2013-2023)", style={'textAlign': 'center', 'color': '#2a3f5f'}),
    html.Label("Select State(s):", style={'fontSize': '18px', 'marginTop': '20px'}),
    dcc.Dropdown(
        id='state-dropdown',
        options=[
            {'label': state, 'value': state} for state in data_avg['state'].unique()
        ],
        multi=True,
        value=[],  # Default is no states selected
        placeholder="Select a state",
        style={'width': '60%', 'margin': '0 auto', 'padding': '10px'}
    ),
    dcc.Graph(id='usage-graph', style={'height': '600px'}),
    dcc.Graph(id='scatter-plot', style={'height': '600px', 'marginTop': '50px'})
])

# Callback to update the graphs based on selected state(s)
@app.callback(
    [Output('usage-graph', 'figure'),
     Output('scatter-plot', 'figure')],
    Input('state-dropdown', 'value')
)
def update_graph(selected_states):
    if not selected_states:
        filtered_data = data_avg[(data_avg['year'] >= 2013) & (data_avg['year'] <= 2023)]
        filtered_prediction = predict_df[(predict_df['year'] >= 2013) & (predict_df['year'] <= 2023)]
    else:
        filtered_data = data_avg[data_avg['state'].isin(selected_states)]
        filtered_data = filtered_data[(filtered_data['year'] >= 2013) & (filtered_data['year'] <= 2023)]
        filtered_prediction = predict_df[predict_df['state'].isin(selected_states)]
        filtered_prediction = filtered_prediction[(filtered_prediction['year'] >= 2013) & (filtered_prediction['year'] <= 2023)]
    
    filtered_data['month_name'] = filtered_data['month'].apply(lambda x: calendar.month_abbr[x])
    filtered_data['year_month'] = filtered_data['year'].astype(str) + '-' + filtered_data['month_name']
    filtered_prediction['month_name'] = filtered_prediction['month'].apply(lambda x: calendar.month_abbr[x])
    filtered_prediction['year_month'] = filtered_prediction['year'].astype(str) + '-' + filtered_prediction['month_name']
    
    # Line plot for actual vs predicted electricity usage
    fig = px.line(
        filtered_data,
        x='year_month',
        y='sales',
        color='state',
        title='Actual vs Predicted Electricity Usage (Sales) from 2013 to 2023 by State and Month',
        markers=True
    )
    
    for state in filtered_prediction['state'].unique():
        state_prediction = filtered_prediction[filtered_prediction['state'] == state]
        fig.add_scatter(x=state_prediction['year_month'], y=state_prediction['predicted_sales'], mode='lines+markers', name=f'{state} (Predicted)')
    
    fig.update_layout(
        xaxis_title='Year-Month',
        yaxis_title='Electricity Usage (Sales)',
        template='plotly_white',
        
        legend_title_text='State',
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False)
    )
    
    # Scatter plot for predicted vs actual values
    merged_df = pd.merge(filtered_data, filtered_prediction, on=['year', 'month', 'state'], suffixes=('_actual', '_predicted'))
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=merged_df['sales'],
        y=merged_df['predicted_sales'],
        mode='markers',
        marker=dict(color='red'),
        name='Predicted vs Actual'
    ))
    scatter_fig.add_trace(go.Scatter(
        x=[merged_df['sales'].min(), merged_df['sales'].max()],
        y=[merged_df['sales'].min(), merged_df['sales'].max()],
        mode='lines',
        line=dict(dash='dash', color='black'),
        name='Perfect Prediction'
    ))
    scatter_fig.update_layout(
        title='Predicted vs Actual Electricity Usage',
        xaxis_title='Actual Sales',
        yaxis_title='Predicted Sales',
        template='plotly_white',
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False)
    )
    
    return fig, scatter_fig

# Run the app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))  # Default to 8050 if PORT is not set
    app.run_server(host='0.0.0.0', port=port, debug=True)
