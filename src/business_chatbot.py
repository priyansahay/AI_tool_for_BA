from src.custom_exception import log_exception
from src.rag_pipeline import load_embedding_model, load_vector_store, retrive_context, build_context

def build_chat_prompt (user_question, context):
    prompt = f""" 
You're an AI Business Intelligence Model.
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

def retrieve_business_context(question, top_k=3):
    embedding_model = load_embedding_model()
    index, documents = load_vector_store()
    retrieved_docs = retrive_context(query=question, embedding_model=embedding_model,index=index, documents=documents, top_k=top_k)
    context = build_context(retrieved_docs)
    return context

def ask_question(question, gemini_client):
    context = (retrieve_business_context(question))
    prompt = (build_chat_prompt(user_question=question, context=context))
    answer = generate_chat_response(prompt=prompt, client=gemini_client)
    return {
        "question": question,
        "context": context,
        "prompt": prompt,
        "answer":answer
    }
