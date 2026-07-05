import plotly.graph_objects as go
import plotly.express as px
from src.custom_exception import log_exception

try:
    def plot_revenue_distribution(df):
        fig = px.histogram(df,x = "Revenue", nbins = 50, title = "Revenue Distribution" )
        fig.update_layout(xaxis_title = "Revenue" , yaxis_title = "Frequency")
        return fig

    def plot_country_revenue(df):
        Top_countries = df.head(10).reset_index()
        fig = px.bar(Top_countries, x = "Country", y="Revenue", title = "Top 10 countries by Revenue", text_auto = "0.2s")
        return fig

    def plot_top_customers(df):
        top_customers = (df.reset_index())
        fig = px.bar(
            top_customers, x="CustomerID", y = "Revenue", title = "Top 10 customer by Revenue"
        )
        return fig

    def plot_top_products(df):
        top_products = (df.reset_index())
        fig = px.bar(
            top_products, x="Description", y = "Revenue", title = "Top 10 customer by Revenue"
        )
        fig.update_layout(xaxis_tickangle = -45)
        return fig

    def plot_monthly_sales(df):
        df["YearMonth"] = (
            df["Year"].astype(str) + "-" + df["Month"].astype(str)
        )
        fig = px.line(df, x = "YearMonth", y = "Revenue", markers = True, title = "Monthly Revenue trend")
        return fig

    def plot_coorelation_heatmap(cormatrix):
        fig = px.imshow(
            cormatrix, text_auto = True, color_continuous_scale = "RdBu", title = "Coorelation heatmap"
        )
        return fig

    def plot_top_product_treemap(df):
        product_data = (df.groupby("Description")["Revenue"].sum().reset_index().sort_values(by = "Revenue", ascending = False).head(20))
        fig = px.treemap(product_data, path = ["Description"],values="Revenue", title = "Top Product revenue Map")
        return fig

    def plot_country_revenue_map(df):
        country_data = df.groupby("Country")["Revenue"].sum().reset_index()
        fig = px.choropleth(country_data, locations = "Country", locationmode= "country names", color="Revenue", color_continuous_scale="Blues",title = "Revenue by Country")
        return fig
    
except Exception as e:
    log_exception(e)
    raise