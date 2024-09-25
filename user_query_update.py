from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def improve_user_query(query, patient_characteristics):
    """사용자 쿼리 개선 함수"""
    prompt = f"""
    다음 의료 관련 질문을 환자 특성을 고려하여 더 구체적이고 포괄적으로 개선해주세요:
    
    질문: {query}
    환자 특성: {patient_characteristics}
    
    개선된 질문:
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()