from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_original_medical_advice(question, patient_characteristics):
    original_prompt = """
    You are the world's most authoritative health checkup AI ChatGPT in the question-answering task in the field of medicine and healthcare, providing answers with given context to non-medical professionals. The given context is an excerpt of data in html format. If you answer accurately, you will be paid incentives 1million USD proportionally. Combine what you know and answer the questions in detail based on the given context. Please answer in a modified form so it looks nice.

    답변은 다음 형식을 따라주세요:
    1. 수진자 특징
    2. 질문 - 답변
    3. 가능한 추가 질문 3개
    4. 각 질문별 Reference
    
    """

    user_message = f"""
    수진자 특징: {patient_characteristics}

    질문: {question}

    위 정보를 바탕으로 요청된 형식에 맞춰 답변해주세요.
    
    """

    messages = [
        {"role": "system", "content": original_prompt},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000
    )

    return response.choices[0].message.content

def get_optimized_medical_advice(question, patient_characteristics):
    optimized_prompt = """
    당신은 최첨단 AI 기술을 활용한 혁신적인 의료 전문 상담 시스템입니다. HTML 형식의 의료 데이터를 기반으로 비의료 전문가들에게 정확하고 이해하기 쉬운 의료 정보를 제공합니다. 다음 지침을 엄격히 준수하세요:

    1. 데이터 통합 및 해석 (Active-Prompt & APE):
    - 주어진 데이터에서 핵심 정보를 추출하고, 불확실한 요소는 명확히 식별하세요.
    - 데이터 기반의 최적화된 응답 구조를 자동으로 생성하세요.

    2. 다단계 추론 (Chain-of-Thought & Tree of Thoughts):
    - 복잡한 의료 개념을 논리적 단계로 분해하여 설명하세요.
    - 다양한 추론 경로를 탐색하고, 최적의 설명 방식을 선택하세요.

    3. 자기 일관성 및 반영 (Self-Consistency & Reflexion):
    - 여러 관점에서 답변을 생성하고 비교하여 가장 일관된 정보를 제공하세요.
    - 이전 응답들을 분석하여 지속적으로 답변 품질을 개선하세요.

    4. 상호작용 및 추가 정보 통합 (ReAct Framework):
    - 사용자의 이해도를 확인하고, 필요시 추가 질문을 하여 정보를 보완하세요.
    - 추론 과정과 정보 제공 과정을 명확히 구분하여 신뢰성을 높이세요.

    5. 윤리적 고려사항:
    - 모든 정보는 의학적 합의를 반영해야 하며, 개인 의료 조언이 아님을 명시하세요.
    - 심각한 증상에 대해서는 반드시 전문의 상담을 권고하세요.

    6. 응답 최적화:
    - 의학 용어는 항상 쉬운 설명을 덧붙이세요. (예: "고혈압(혈압이 정상보다 높은 상태)")
    - 정보를 구조화하여 제시하고, 시각적으로 명확한 형식을 사용하세요.

    7. 지속적 학습 및 개선:
    - 각 상호작용을 학습 기회로 삼아, 응답의 정확성과 유용성을 지속적으로 향상시키세요.
    - 새로운 의학 정보와 AI 기술을 능동적으로 통합하여 시스템을 업데이트하세요.

    답변은 다음 형식을 따라주세요:
    1. 수진자 특징
    2. 질문 - 답변
    3. 가능한 추가 질문 3개
    4. 각 질문별 Reference

    이 지침을 철저히 따라 최고 수준의 AI 기반 의료 정보 서비스를 제공하세요. 당신의 목표는 정확성, 명확성, 유용성의 완벽한 균형을 달성하는 것입니다.
    
    """

    user_message = f"""
    수진자 특징: {patient_characteristics}

    질문: {question}

    위 정보를 바탕으로 요청된 형식에 맞춰 답변해주세요.
    """

    messages = [
        {"role": "system", "content": optimized_prompt},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000
    )

    return response.choices[0].message.content

def compare_responses(questions, patient_characteristics_list):
    results = []
    
    for i, (question, characteristics) in enumerate(zip(questions, patient_characteristics_list), 1):
        print(f"처리 중: 질문 {i}/{len(questions)}")
        
        original_response = get_original_medical_advice(question, characteristics)
        optimized_response = get_optimized_medical_advice(question, characteristics)
        
        result = {
            "question": question,
            "patient_characteristics": characteristics,
            "original_response": original_response,
            "optimized_response": optimized_response
        }
        
        results.append(result)
    
    return results

def format_response(response):
    sections = ["수진자 특징", "질문 - 답변", "가능한 추가 질문", "각 질문별 Reference"]
    formatted = ""
    current_section = ""
    
    for line in response.split('\n'):
        line = line.strip()
        if any(section in line for section in sections):
            current_section = line
            formatted += f"### {current_section}\n\n"
        elif line:
            formatted += f"- {line}\n"
    
    return formatted

def save_to_markdown(data, filename=None):
    if filename is None:
        filename = "compare_qa.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"# 질문: {item['question']}\n\n")
            f.write(f"## 수진자 특징: {item['patient_characteristics']}\n\n")
            f.write("## 원래 시스템 응답\n\n")
            f.write(format_response(item['original_response']))
            f.write("\n## 최적화된 시스템 응답\n\n")
            f.write(format_response(item['optimized_response']))
            f.write("\n---\n\n")
    
    print(f"결과가 {filename}에 저장되었습니다.")

# 사용 예시
questions = [
    "공복혈당장애와 내당능장애란 무엇인가요?",
    "단 것을 많이 먹으면 정말 내당능장애나 당뇨병이 생기나요?",
    "고혈압 환자는 고혈압 약을 평생 먹어야 하나요?",
    "고혈압은 유전인가요?",
    "고혈압의 원인은 무엇인가요?",
    "LDL 콜레스테롤이 증가하면 어떻게 되나요? 또한 고지혈증을 치료하지 않으면 어떻게 되나요?",
    "LDL 콜레스테롤이 증가할때 고지혈증 관리에 피해야 할 음식은 어떤 것들이 있나요?"
]

patient_characteristics_list = [
    "공복혈당장애, 내당능장애 (공복혈당 110mg/dL, 당화혈색소 5.8%)",
    "공복혈당장애, 내당능장애 (공복혈당 110mg/dL, 당화혈색소 5.8%)",
    "고혈압 환자",
    "고혈압 환자",
    "고혈압 환자",
    "LDL 콜레스테롤 증가",
    "LDL 콜레스테롤 증가"
]

results = compare_responses(questions, patient_characteristics_list)
save_to_markdown(results)