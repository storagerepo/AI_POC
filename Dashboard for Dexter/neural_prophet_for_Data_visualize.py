
###Neural Prophet

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# Streamlit configuration for light theme
st.set_page_config(page_title="Real Estate Insights Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load dataset
@st.cache_data
def load_data(file_path="time_series.csv"):
    df = pd.read_csv(file_path, parse_dates=["Timestamp"])
    return df

# Load property prices
@st.cache_data
def load_property_prices(file_path="p.csv"):
    property_prices = pd.read_csv(file_path)
    return property_prices

from neuralprophet import NeuralProphet
import pandas as pd
import plotly.graph_objects as go

from neuralprophet import NeuralProphet
import pandas as pd
import plotly.graph_objects as go




def forecast_selling_rates(df):
    # Filter "Sold" data
    sold_data = df[df["Interaction Type"] == "Sold"]
    sold_data["Date"] = sold_data["Timestamp"].dt.date
    sales_per_day = sold_data.groupby("Date").size().reset_index(name="y")
    sales_per_day.rename(columns={"Date": "ds"}, inplace=True)
    
    # Initialize NeuralProphet model
    model = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True    )
    model.fit(sales_per_day, freq="D")

    # Forecast future data (removing the freq argument)
    future = model.make_future_dataframe(sales_per_day, periods=30)  # Daily forecast for 30 days
    forecast = model.predict(future)
    
    # Plot forecast
    fig = go.Figure()
    # Actual data
    fig.add_trace(go.Scatter(
        x=sales_per_day["ds"],
        y=sales_per_day["y"],
        mode="lines+markers",
        name="Actual Sales",
        hoverinfo="x+y"
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat1"],  # NeuralProphet output column for the forecast
        mode="lines",
        name="Forecasted Sales",
        hoverinfo="x+y",
        line=dict(dash="dash")
    ))
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat1"],
        mode="lines",
        line=dict(width=0.5, color="lightgrey"),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat1"],
        mode="lines",
        line=dict(width=0.5, color="lightgrey"),
        showlegend=False,
        fill="tonexty",
        fillcolor="rgba(173,216,230,0.3)"
    ))
    
    # Add title and annotations
    fig.update_layout(
        title="Daily Property Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Number of Properties Sold",
        hovermode="x",
        template="plotly_white",
    )
    return fig

# User Age vs Property Preferences
def user_age_vs_preferences(df):
    age_bins = [15, 20, 25, 30, 35, 40, 50, 60]
    age_labels = ["15-20", "20-25", "25-30", "30-35", "35-40", "40-50", "50-60"]
    df["Age Range"] = pd.cut(df["User Age"], bins=age_bins, labels=age_labels, right=False)

    sold_bid_data = df[df["Interaction Type"].isin(["view", "like", "shared", "favourites", "Sold", "bid", "searched for"])]
    preference_data = sold_bid_data.groupby(["Age Range", "Property Type"]).size().reset_index(name="Count")

    # Ensure all property types are represented in each age range
    all_combinations = pd.MultiIndex.from_product([
        preference_data["Age Range"].unique(),
        preference_data["Property Type"].unique()
    ], names=["Age Range", "Property Type"])
    preference_data = preference_data.set_index(["Age Range", "Property Type"]).reindex(all_combinations, fill_value=0).reset_index()

    fig = px.bar(preference_data, x="Age Range", y="Count", color="Property Type", barmode="group", title="User Age vs Property Preferences")
    fig.update_layout(template="plotly_white", xaxis_title="Age Range", yaxis_title="Property Preferences Count")
    return fig

# Investment Opportunities
def investment_opportunities(df):
    sold_data = df[df["Interaction Type"] == "Sold"]
    area_sales = sold_data.groupby("State").size().reset_index(name="Sales")
    
    fig = px.bar(area_sales, x="State", y="Sales", title="Investment Opportunities by Area", text="Sales")
    fig.update_layout(template="plotly_white", xaxis_title="State", yaxis_title="Number of Sales")
    fig.update_traces(textposition="outside")
    return fig

# User Interest Analysis
def user_interest_analysis(df):
    interaction_data = df.groupby(["State", "Property Type", "Interaction Type"]).size().reset_index(name="Count")
    fig = px.sunburst(
        interaction_data,
        path=["State", "Property Type", "Interaction Type"],
        values="Count",
        title="User Interest Analysis",
        color="State",
        color_discrete_sequence=px.colors.qualitative.Bold,  # Use a vibrant color scheme
    )
    fig.update_layout(
        template="plotly_dark",  
        height=800, 
        width=800,  
        margin=dict(t=50, l=10, r=10, b=10), 
    )
    return fig

# User Engagement by Property Type
def user_engagement_by_property_type(df):
    engagement_data = df.groupby(["Property Type", "Interaction Type"]).size().reset_index(name="Count")
    
    fig = px.bar(engagement_data, x="Property Type", y="Count", color="Interaction Type", 
                 title="User Engagement by Property Type", barmode="stack")
    fig.update_layout(template="plotly_white", xaxis_title="Property Type", yaxis_title="Engagement Count")
    return fig

# Seasonal User Engagement Trends
def seasonal_user_engagement(df):
    df["Month"] = df["Timestamp"].dt.month
    engagement_by_month = df.groupby("Month").size().reset_index(name="Count")
    
    fig = px.bar(engagement_by_month, x="Month", y="Count", title="Seasonal User Engagement Trends")
    fig.update_layout(template="plotly_white", xaxis_title="Month", yaxis_title="User Engagement Count")
    return fig

# Top Property Types in States
def top_property_types_in_states(df):
    property_type_data = df.groupby(["State", "Property Type"]).size().reset_index(name="Count")
    
    fig = px.bar(property_type_data, x="State", y="Count", color="Property Type", 
                 title="Top Property Types in States(Total Viewed)", barmode="group")
    fig.update_layout(template="plotly_white", xaxis_title="State", yaxis_title="Number of Properties")
    return fig
# Simple line graph of sales over time
def sales_over_time(df):
    # Filter sold data
    sold_data = df[df["Interaction Type"] == "Sold"]
    sold_data["Date"] = sold_data["Timestamp"].dt.date
    sales_per_day = sold_data.groupby("Date").size().reset_index(name="Sales")
    
    # Create a simple line graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sales_per_day["Date"],
        y=sales_per_day["Sales"],
        mode="lines+markers",
        name="Properties Sold",
        hoverinfo="x+y"
    ))
    
    # Title and axis labels
    fig.update_layout(
        title="Properties Sold Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Properties Sold",
        template="plotly_white"
    )
    return fig

# Calculate Total Revenue
def calculate_total_revenue(df, property_prices):
    sold_data = df[df["Interaction Type"] == "Sold"]
    sold_data = sold_data.merge(property_prices, on="Property ID", how="left")
    sold_data["Revenue"] = sold_data["Price"]
    total_revenue = sold_data["Revenue"].sum()

    # Visualization
    revenue_per_property = sold_data.groupby("Property Type")["Revenue"].sum().reset_index()
    fig = px.bar(revenue_per_property, x="Property Type", y="Revenue", title="Total Revenue by Property Type", text="Revenue")
    fig.update_layout(template="plotly_white", xaxis_title="Property Type", yaxis_title="Revenue")
    return fig, total_revenue

# Calculate Conversion Rate
def calculate_conversion_rate(df):
    inquiry_data = df[df["Interaction Type"].isin(["view", "like", "shared", "favourites"])]
    sold_or_bid_data = df[df["Interaction Type"].isin(["Sold", "bid"])]

    total_inquiries = inquiry_data.shape[0]
    total_sales_or_bids = sold_or_bid_data.shape[0]

    conversion_rate = (total_sales_or_bids / total_inquiries) * 100 if total_inquiries > 0 else 0

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Inquiries"], y=[total_inquiries], name="Inquiries"))
    fig.add_trace(go.Bar(x=["Sales or Bids"], y=[total_sales_or_bids], name="Sales or Bids"))
    fig.update_layout(
        title="Conversion Rate: Inquiries to Sales/Bids",
        xaxis_title="Category",
        yaxis_title="Count",
        barmode="group",
        template="plotly_white",
    )
    return fig, conversion_rate

# Streamlit App with the new options
st.title("Just Keys Insights Dashboard")

# Load datasets
df = load_data()
property_prices = load_property_prices()

# Task Selection
analysis_options = st.sidebar.multiselect("Select Analyses to Display", 
    [
        "Forecast Selling Rates", 
        "User Age with Property type", 
        "Investment Opportunities", 
        "User Interest Analysis",
        "User Engagement by Property Type",
        "Seasonal User Engagement Trends",
        "Top Property Types in States(Total Viewed)",
        "Total Revenue by Property Type",
        "Conversion Rate: Inquiries to Sales/Bids",
    ])

if "Forecast Selling Rates" in analysis_options:
    st.header("Property Sales Forecast")
    st.plotly_chart(forecast_selling_rates(df), use_container_width=True)

if "User Age with Property type" in analysis_options:
    st.header("User Age with Property type")
    st.plotly_chart(user_age_vs_preferences(df), use_container_width=True)

if "Investment Opportunities" in analysis_options:
    st.header("Investment Opportunities")
    st.plotly_chart(investment_opportunities(df), use_container_width=True)

if "User Interest Analysis" in analysis_options:
    st.header("User Interest Analysis")
    st.plotly_chart(user_interest_analysis(df), use_container_width=True)

if "User Engagement by Property Type" in analysis_options:
    st.header("User Engagement by Property Type")
    st.plotly_chart(user_engagement_by_property_type(df), use_container_width=True)

if "Seasonal User Engagement Trends" in analysis_options:
    st.header("Seasonal User Engagement Trends")
    st.plotly_chart(seasonal_user_engagement(df), use_container_width=True)

if "Top Property Types in States(Total Viewed)" in analysis_options:
    st.header("Top Property Types in States(Total Viewed)")
    st.plotly_chart(top_property_types_in_states(df), use_container_width=True)


if "Total Revenue by Property Type" in analysis_options:
    st.header("Total Revenue by Property Type")
    revenue_fig, total_revenue = calculate_total_revenue(df, property_prices)
    st.plotly_chart(revenue_fig, use_container_width=True)
    st.write(f"Total Revenue: ${total_revenue:,.2f}")

if "Conversion Rate: Inquiries to Sales/Bids" in analysis_options:
    st.header("Conversion Rate: Inquiries to Sales/Bids")
    conversion_fig, conversion_rate = calculate_conversion_rate(df)
    st.plotly_chart(conversion_fig, use_container_width=True)
    st.write(f"Conversion Rate: {conversion_rate:.2f}%")

