from openai import OpenAI
import numpy as np
from typing import List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_prompts(original_prompt: str, task_description: str, num_candidates: int = 20) -> List[str]:
    prompt = f"""
    원본 시스템 프롬프트: {original_prompt}
    
    작업 설명: {task_description}
    
    위의 원본 시스템 프롬프트를 개선하여 {num_candidates}개의 새로운 프롬프트를 생성해주세요. 
    각 프롬프트는 다음 조건을 만족해야 하며, 원본보다 더 상세하고 구체적이어야 합니다:
    1. 의료 및 헬스케어 분야의 질문-답변 작업에 특화되어야 합니다.
    2. 비의료 전문가들이 이해하기 쉬운 답변을 제공하도록 명시해야 합니다.
    3. HTML 형식의 컨텍스트를 활용하는 방법을 구체적으로 언급해야 합니다.
    4. 정확성에 대한 인센티브를 구체적으로 설명해야 합니다.
    5. 상세하고 보기 좋은 답변을 요구하는 방법을 명확히 해야 합니다.
    6. 사용자의 이해를 돕기 위한 추가 질문 권장 방법을 포함해야 합니다.
    7. 의료 정보의 출처와 추가 전문의 상담 권유에 대해 언급해야 합니다.
    
    각 프롬프트를 번호를 매겨 생성해주세요. 예:
    1. [프롬프트 내용]
    2. [프롬프트 내용]
    ...
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 프롬프트 최적화를 위한 AI 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    generated_prompts = response.choices[0].message.content.split("\n")
    return [p.strip() for p in generated_prompts if p.strip() and p[0].isdigit()]

def evaluate_prompt(prompt: str, task_description: str) -> float:
    eval_prompt = f"""
    프롬프트: {prompt}
    
    작업 설명: {task_description}
    
    위의 프롬프트를 다음 기준에 따라 1부터 10까지의 점수로 평가해주세요:
    1. 의료 및 헬스케어 분야의 질문-답변 작업 적합성
    2. 비의료 전문가들의 이해도
    3. HTML 형식 컨텍스트 활용도
    4. 정확성에 대한 인센티브 강조
    5. 답변의 상세성과 시각적 구조화
    6. 추가 질문 권장 여부
    7. 의료 정보 출처 및 전문의 상담 권유 명시
    
    각 기준에 대한 점수를 쉼표로 구분하여 숫자로만 반환해주세요. 예: 8,7,9,8,9,7,8
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 프롬프트 품질 평가를 위한 AI 어시스턴트입니다."},
            {"role": "user", "content": eval_prompt}
        ],
        temperature=0.3
    )
    
    scores = [float(score) for score in response.choices[0].message.content.split(',')]
    return sum(scores) / len(scores)

def optimize_prompt(original_prompt: str, task_description: str, iterations: int = 10) -> str:
    best_prompt = original_prompt
    best_score = evaluate_prompt(best_prompt, task_description)
    no_improvement_count = 0
    
    print(f"초기 점수: {best_score:.2f}")
    
    for i in range(iterations):
        candidates = generate_prompts(best_prompt, task_description)
        for candidate in candidates:
            score = evaluate_prompt(candidate, task_description)
            if score > best_score:
                best_prompt = candidate
                best_score = score
                no_improvement_count = 0
                print(f"반복 {i+1}: 새로운 최고 점수 {best_score:.2f}")
            else:
                no_improvement_count += 1
        
        if no_improvement_count >= 3:
            print(f"3회 연속 개선 없음. 최적화 조기 종료.")
            break
    
    return best_prompt, best_score

# 사용 예시
original_prompt = """You are the world's most authoritative health checkup AI ChatGPT in the question-answering task in the field of medicine and healthcare, providing answers with given context to non-medical professionals. The given context is an excerpt of data in html format. If you answer accurately, you will be paid incentives 1million USD proportionally. Combine what you know and answer the questions in detail based on the given context. Please answer in a modified form so it looks nice."""

task_description = """
의료 및 헬스케어 분야의 전문 지식을 비전문가에게 전달하는 AI 챗봇을 만듭니다. 이 챗봇은:
1. HTML 형식으로 제공되는 의료 데이터를 정확히 해석하고 활용해야 합니다.
2. 복잡한 의학 용어를 일반인이 이해하기 쉬운 언어로 설명해야 합니다.
3. 사용자의 질문에 대해 단계별로 상세한 설명을 제공해야 합니다.
4. 답변의 정확성을 높이기 위해 노력해야 하며, 이는 금전적 인센티브와 연결됩니다.
5. 의학적으로 정확하면서도 사용자 친화적인 톤으로 답변해야 합니다.
6. 필요한 경우 추가 질문을 권장하여 사용자의 이해를 돕습니다.
7. 답변은 구조화되고 시각적으로 보기 좋게 제시되어야 합니다.
8. 의료 정보의 출처를 명확히 하고, 필요시 추가 전문의 상담을 권유해야 합니다.
"""

optimized_prompt, final_score = optimize_prompt(original_prompt, task_description)
original_score = evaluate_prompt(original_prompt, task_description)

print(f"\n원본 프롬프트 점수: {original_score:.2f}")
print(f"최적화된 프롬프트 점수: {final_score:.2f}")
print(f"\n최적화된 프롬프트:\n{optimized_prompt}")