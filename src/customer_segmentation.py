from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
import joblib
from src.custom_exception import log_exception

try:
    def scale_features(rfm_df):
        features = rfm_df[["Recency","Frequency","Monetary"]]
        scaler = StandardScaler()
        scaled_func = scaler.fit_transform(features)
        return scaled_func, scaler
    
    def finding_optimal_clusters(scaled_func, maxcluster=10):
        inertia=[]
        cluster_range = range(2, maxcluster + 1)
        for i in cluster_range:
            model = KMeans(
                n_clusters= i,
                random_state = 42,
                n_init = 10
            )
            model.fit(scale_features)
            inertia.append(model.inertia_)
        elbow_df = pd.DataFrame({
            "Clusters" : list(cluster_range),
            "WCSS": inertia
        }) 
        return elbow_df
    
    def plot_elbow(elbow_df):
        fig= px.line(elbow_df, x = "Clusters", y= "WCSS", markers = True, title = "Elbow Graph")
        return fig
    
    def train_segmentation_model(scaled_features, n_clusters = 4):
        model = KMeans(
            n_clusters= n_clusters,
            random_state=42,
            n_init=10
        )
        model.fit_transform(scale_features)
        return model
    
    def assign_clusters(rfm_df, model, scaled_features):
        segmented_df = rfm_df.copy()
        segmented_df["Cluster"] = (model.predict(scaled_features))
        return segmented_df
    
    def cluster_summary(df):
        summary = (df.groupby("Cluster").agg(
            Customer_Count = ("CustomerID","count"),
            Avg_Recency = ("Recency","mean"),
            Avg_Frequency = ("Frequency","mean"),
            Avg_Monetary = ("Monetary","mean")
        ).round(2).reset_index())
        return summary
    
    # ADDING BUSINESS SEGMENTS
    def label_customer_segments(df):
        segment_summary = (df.groupby("Cluster")["Monetary"].mean().sort_values(ascending = False))
        ordered_clusters = (segment_summary.index.tolist())
        labels={}
        if len(ordered_clusters)>=4:
            labels[ordered_clusters[0]]="VIP"
            labels[ordered_clusters[1]]="Loyal"
            labels[ordered_clusters[2]]="Regular"
            labels[ordered_clusters[3]]="At Risk"
        else:
            for i in ordered_clusters:
                labels[i]=f"Segment_{i}"
        df["Segment"]=(df["Cluster"].map(labels))
        return df
    
    def save_model(model,file_path):
        Path(file_path).parent.mkdir(parents =True, exist_ok=True)
        joblib.dump(model,file_path)

    def save_segmented_data(segmented_df,file_path):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        segmented_df.to_csv(file_path,index=False)

    def segmentation_pipeline(rfm_df,n_clusters=4):
        scaled_features,scaler = (scale_features(rfm_df))
        model = (train_segmentation_model(scaled_features,n_clusters))
        segmented_df = (assign_clusters(rfm_df,model,scaled_features))
        segmented_df = (label_customer_segments(segmented_df))
        summary = (cluster_summary(segmented_df))
        return {
            "segmented_df": segmented_df,
            "summary": summary,
            "model":model,
            "scaler":scaler
        }
    
except Exception as e:
    log_exception(e)
    raise