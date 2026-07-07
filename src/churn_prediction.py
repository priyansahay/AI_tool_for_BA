import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
from src.custom_exception import log_exception

try:

    def prepare_churn_data(churn_df):
        df = churn_df.copy()
        if "CustomerID" in df.columns:
            df = df.drop(columns = ["CustomerID"])
        if "Segment" in df.columns:
            df = pd.get_dummies(df, columns=["Segment"], drop_first = True)
        return df

    def split_features_target(df):
        x = df.drop(columns = ["Churn"])
        y = df["Churn"]
        return x,y

    def split_train_test(x,y):
        x_train, x_test, y_train, y_test = (train_test_split(x,y, test_size=0.2, random_state=42, stratify=y))
        return x_train, x_test, y_train, y_test

    def train_churn_model(x_train, y_train):
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(x_train,y_train)
        return model

    def evualate_churn_model(model, x_test, y_test):
        prediction = model.predict(x_test)
        metrics = {
            "Accuracy" : round(accuracy_score(y_test,prediction),4),
            "Precision": round(precision_score(y_test,prediction),4),
            "Recall": round(recall_score(y_test,prediction)),
            "F1":round(f1_score(y_test,prediction),4)
        }
        cm = confusion_matrix(y_test,prediction)
        return prediction, metrics, cm

    def get_feature_importance(model,x_train):
        importance_df = pd.DataFrame(
            {"Feature":x_train.columns,
            "Importance": model. feature_importances_
            })
        importance_df = (
            importance_df.sort_values(by="Importance", ascending=False)
        )
        return importance_df

    def generate_churn_score(model,x):
        churn_prob = (model.predict_proba(x))[:,1]
        prediction_df = x.copy()
        prediction_df["Churn_Probability"] = churn_prob
        prediction_df["Risk_level"] = prediction_df["Churn_Probability"].apply(assign_risk_level)
        return prediction_df

    def assign_risk_level(score):
        if score >=0.75:
            return "High Risk"
        elif score >= 0.40:
            return "Medium Risk"
        return "Low Risk" 

    def save_churn_model(model, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model,output_path)

    def save_churn_prediction(prediction_df,output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(output_path,index=False)
        
    def churn_prediction_pipeline(churn_df):
        prepared_df = (prepare_churn_data(churn_df))
        x,y = (split_features_target(prepared_df))
        x_train,x_test, y_train,y_test = (split_train_test(x,y))
        model = train_churn_model(x_train,y_train)
        (prediction, metrics, confusion_mat) = evualate_churn_model(model, x_test, y_test)
        feature_importance = get_feature_importance(model, x_train)
        churn_score = generate_churn_score(model,x)
        return {
            "model":model,
            "metrics": metrics,
            "confusion_matrix": confusion_mat,
            "feature_importance": feature_importance,
            "churn_scores":churn_score
        }
except Exception as e:
    log_exception(e)
    raise