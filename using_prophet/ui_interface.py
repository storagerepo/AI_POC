import dash
from dash import dcc, html
import dash.dash_table as dash_table
import pandas as pd
import plotly.graph_objs as go


historical_df = pd.read_csv('./dataset/mock_time_series_property_data.csv')
historical_df = historical_df[historical_df['property_id'] == 'property_1']
historical_df['date'] = pd.to_datetime(historical_df['date'])

predicted_df = pd.read_csv('./predicted_dataset/predicted_prices.csv')
predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])

historical_df['date'] = historical_df['date'].dt.strftime('%Y %b')
predicted_df['Date'] = predicted_df['Date'].dt.strftime('%Y %b')

historical_df['price'] = historical_df['price'].astype(int)
predicted_df['Predicted Price'] = predicted_df['Predicted Price'].astype(int)
predicted_df['Estimated Low Range'] = predicted_df['Estimated Low Range'].astype(int)
predicted_df['Estimated High Range'] = predicted_df['Estimated High Range'].astype(int)



app = dash.Dash(__name__)

style_table = {
    'margin': 'auto',  
    'width': '60%',    
}

app.layout = html.Div([
    
    html.H2('Property Forecast Table', style={'textAlign': 'left'}),
    
    html.H4('Historical Data (2014 OCT -2024 JULY)', style={'textAlign': 'center'}),
    dash_table.DataTable(
        data=historical_df[['date', 'price']].rename(columns={'date': 'Date', 'price': 'Historical Price'}).to_dict('records'),
        columns=[{"name": i, "id": i} for i in ['Date', 'Historical Price']],
        style_table=style_table,
        style_cell={'textAlign': 'center'},  
        style_data_conditional=[
            {'if': {'column_id': 'Date'}, 'textAlign': 'left'},  
        ],
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        sort_action="native", 
        sort_mode="multi",
        page_size=10 
    ),

    html.H4('Predicted Data (2024 Oct - 2025 July)', style={'textAlign': 'center'}),
    dash_table.DataTable(
        data=predicted_df[['Date', 'Predicted Price','Estimated Low Range','Estimated High Range']].to_dict('records'),
        columns=[{"name": i, "id": i} for i in ['Date', 'Predicted Price','Estimated Low Range','Estimated High Range']],
        style_table=style_table,
        style_cell={'textAlign': 'center'},  
        style_data_conditional=[
            {
                'if': {'column_id': 'Predicted Price'}, 
                'backgroundColor': 'lightgreen',  
                'color': 'black',
            }
        ],
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        sort_action="native",
        sort_mode="multi",
        page_size=10  
    ),
    html.Div('Note: Green points indicate predicted values.', style={'textAlign': 'center'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
