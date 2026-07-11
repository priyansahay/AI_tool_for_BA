from src.custom_exception import log_exception
from src.rag_pipeline import load_embedding_model, load_vector_store, retrive_context, build_context
try:

    def build_chat_prompt (user_question, context):
        prompt = f""" 
    You're an AI Business Intellegence Model.
    Use only the context below to answer
    CONTEXT:{context},
    QUESTION:{user_question}
    Provide: 1. Direct Answer
             2. Business Insight
             3. Recommended Action 
    """
        return prompt
    
    def generate_chat_response(prompt, client):
        response = client.models.generate_content(model = "gemini-2.5-flash", contents = prompt)
        answer = response.text
        return answer