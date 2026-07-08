from src.custom_exception import log_exception
from pathlib import Path
from google import genai
import datetime
def create_business_summary(next_day_forecast, segmented_df, churn_scores,anomaly_results):
    metrics= {
        "rev": round(next_day_forecast,2),
        "vip": segmented_df[segmented_df["Segment"]=="VIP"].shape[0],
        "loyal": segmented_df[segmented_df["Segment"]=="Loyal"].shape[0],
        "regular":segmented_df[segmented_df["Segment"]=="Regular"].shape[0],
        "at_risk": segmented_df[segmented_df["Segment"]=="At Risk"].shape[0],
        "high_churn": churn_scores[churn_scores["Risk_Level"]=="High Risk"].shape[0],
        "medium_churn": churn_scores[churn_scores["Risk_Level"]=="Medium Risk"].shape[0],
        "anomalies": anomaly_results[anomaly_results["Anomaly"]=="Anomaly"].shape[0]
    }
    return metrics

def build_prompt(metrics):
    prompt = f"""
            rev:{metrics['rev']},
            vip:{metrics['vip']},
            loyal:{metrics['loyal']},
            regular:{metrics['regular']},
            atrisk:{metrics['at_risk']},
            highchurn:{metrics['high_churn']},
            anom:{metrics['anomalies']}
            summary + actions
        """
    return prompt

def generate_llm_report(prompt, client = None):
    if client is None:
        raise ValueError("LLM Client not provided")
    response = client.model.generate_content(model = "gemini/gemini-2.5-flash",contents=prompt)
    report = response.text
    return report

def generate_local_report(metrics):
    report = f"""
        ---- Forecast Revenue:----\n
        Rs {metrics['rev']:,.2f}\n

        ---- Customer Segments: ----\n
        VIP Customers: {metrics['vip']}\n
        Loyal Customers: {metrics['loyal']}\n
        Regular Customers: {metrics['regular']}\n
        At-risk Customer: {metrics['at_risk']}\n

        ---- Customer Risk: ----\n
        High Risk: {metrics['high_churn']}\n
        Medium Risk: {metrics['medium_churn']}\n
        Anomalies Detected: {metrics['anomalies']}\n
        """
    return report

def save_report(report, output_dir = "reports"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_name = (f"executive_report_"
                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                   ".txt")
    report_path = (Path(output_dir)/report_name)
    with open(report_path,"w", encoding="utf-8") as file:
        file.write(report)
    return str(report_path)

