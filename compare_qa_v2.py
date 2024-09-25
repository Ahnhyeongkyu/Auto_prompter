import time
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime

# 환경 변수에서 API 키 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 원래의 시스템 프롬프트 정의
ORIGINAL_PROMPT = """
You are the world's most authoritative health checkup AI ChatGPT in the question-answering task in the field of medicine and healthcare, providing answers with given context to non-medical professionals. 
The given context is an excerpt of data in html format. If you answer accurately, you will be paid incentives 1million USD proportionally. 
Combine what you know and answer the questions in detail based on the given context. Please answer in a modified form so it looks nice.

답변은 다음 형식을 따라주세요:
1. 수진자 특징
2. 질문 - 답변
3. 가능한 추가 질문 3개
4. 각 질문별 Reference
"""

# 최적화된 시스템 프롬프트 정의
OPTIMIZED_PROMPT = """
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

def improve_user_query(query, patient_characteristics):
    """
    사용자 쿼리 개선 함수
    
    :param query: 원래 사용자 질문
    :param patient_characteristics: 환자 특성
    :return: 개선된 질문, 사용된 토큰 수, 실행 시간
    """
    start_time = time.time()
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
    
    end_time = time.time()
    return response.choices[0].message.content.strip(), response.usage.total_tokens, end_time - start_time


def get_medical_advice(question, patient_characteristics, system_prompt):
    """
    의료 조언 생성 함수
    
    :param question: 질문 (개선된 질문 또는 원래 질문)
    :param patient_characteristics: 환자 특성
    :param system_prompt: 사용할 시스템 프롬프트
    :return: 생성된 의료 조언, 사용된 토큰 수, 실행 시간
    """
    start_time = time.time()
    user_message = f"""
    수진자 특징: {patient_characteristics}

    질문: {question}

    위 정보를 바탕으로 요청된 형식에 맞춰 답변해주세요.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000
    )

    end_time = time.time()
    return response.choices[0].message.content, response.usage.total_tokens, end_time - start_time

def compare_responses(questions, patient_characteristics_list, improve_queries=True):
    """
    주어진 질문들에 대해 응답을 비교하는 함수
    
    :param questions: 질문 목록
    :param patient_characteristics_list: 환자 특성 목록
    :param improve_queries: 쿼리 개선 여부 (기본값: True)
    :return: 결과 리스트, 총 토큰 수, 총 실행 시간
    """
    
    results = []
    total_tokens = 0
    total_time = 0
    
    for i, (question, characteristics) in enumerate(zip(questions, patient_characteristics_list), 1):
        print(f"처리 중: 질문 {i}/{len(questions)}")
        
        query_result = {
            "original_question": question,
            "patient_characteristics": characteristics,
            "resource_usage": {}
        }
        
        # 쿼리 개선 적용 여부에 따른 처리
        if improve_queries:
            improved_question, improvement_tokens, improvement_time = improve_user_query(question, characteristics)
            query_result["improved_question"] = improved_question
            query_result["resource_usage"]["query_improvement"] = {
                "tokens": improvement_tokens,
                "time": improvement_time
            }
            total_tokens += improvement_tokens
            total_time += improvement_time
            print(f"개선된 질문: {improved_question}")
        else:
            improved_question = question
        
        # 원래 시스템과 최적화된 시스템의 응답 생성
        original_response, original_tokens, original_time = get_medical_advice(improved_question, characteristics, ORIGINAL_PROMPT)
        optimized_response, optimized_tokens, optimized_time = get_medical_advice(improved_question, characteristics, OPTIMIZED_PROMPT)
        
        # 결과 저장
        query_result["original_response"] = original_response
        query_result["optimized_response"] = optimized_response
        query_result["resource_usage"]["original_response"] = {
            "tokens": original_tokens,
            "time": original_time
        }
        query_result["resource_usage"]["optimized_response"] = {
            "tokens": optimized_tokens,
            "time": optimized_time
        }
        
        # 총 자원 사용량 계산
        total_tokens += original_tokens + optimized_tokens
        total_time += original_time + optimized_time
        
        query_result["resource_usage"]["total"] = {
            "tokens": sum(item["tokens"] for item in query_result["resource_usage"].values()),
            "time": sum(item["time"] for item in query_result["resource_usage"].values())
        }
        
        results.append(query_result)
    
    return results, total_tokens, total_time

def format_response(response):
    """
    응답을 마크다운 형식으로 포맷팅하는 함수
    
    :param response: 원본 응답 텍스트
    :return: 포맷팅된 마크다운 텍스트
    """

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
    """
    결과를 마크다운 파일로 저장하는 함수
    
    :param data: 저장할 데이터
    :param filename: 파일 이름 (기본값: None)
    """
    if filename is None:
        filename = f"compare_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"# 원래 질문: {item['original_question']}\n\n")
            if 'improved_question' in item and item['improved_question']:
                f.write(f"## 개선된 질문: {item['improved_question']}\n\n")
            f.write(f"## 수진자 특징: {item['patient_characteristics']}\n\n")
            f.write("## 원래 시스템 응답\n\n")
            f.write(format_response(item['original_response']))
            f.write("\n## 최적화된 시스템 응답\n\n")
            f.write(format_response(item['optimized_response']))
            f.write("\n---\n\n")
    
    print(f"결과가 {filename}에 저장되었습니다.")

def save_detailed_results(results, filename):
    """
    상세 결과를 JSON 파일로 저장하는 함수
    
    :param results: 저장할 결과 데이터
    :param filename: 저장할 파일 이름
    """

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"상세 결과가 {filename}에 저장되었습니다.")

# 메인 실행 부분
if __name__ == "__main__":
    # 질문 목록 정의
    questions = [
        "공복혈당장애와 내당능장애란 무엇인가요?",
        "단 것을 많이 먹으면 정말 내당능장애나 당뇨병이 생기나요?",
        "고혈압 환자는 고혈압 약을 평생 먹어야 하나요?",
        "고혈압은 유전인가요?",
        "고혈압의 원인은 무엇인가요?",
        "LDL 콜레스테롤이 증가하면 어떻게 되나요? 또한 고지혈증을 치료하지 않으면 어떻게 되나요?",
        "LDL 콜레스테롤이 증가할때 고지혈증 관리에 피해야 할 음식은 어떤 것들이 있나요?"
    ]

    # 환자 특성 목록 정의
    patient_characteristics_list = [
        "공복혈당장애, 내당능장애 (공복혈당 110mg/dL, 당화혈색소 5.8%)",
        "공복혈당장애, 내당능장애 (공복혈당 110mg/dL, 당화혈색소 5.8%)",
        "고혈압 환자",
        "고혈압 환자",
        "고혈압 환자",
        "LDL 콜레스테롤 증가",
        "LDL 콜레스테롤 증가"
    ]

    # 쿼리 개선을 적용하여 응답 비교
    results_improved, tokens_improved, time_improved = compare_responses(questions, patient_characteristics_list, improve_queries=True)
    save_to_markdown(results_improved, "compare_qa_with_improved_queries.md")
    save_detailed_results(results_improved, "detailed_results_with_improved_queries.json")

    # 쿼리 개선을 적용하지 않고 응답 비교
    results_original, tokens_original, time_original = compare_responses(questions, patient_characteristics_list, improve_queries=False)
    save_to_markdown(results_original, "compare_qa_without_improved_queries.md")
    save_detailed_results(results_original, "detailed_results_without_improved_queries.json")


    # 전체 자원 사용 비교 데이터 저장
    resource_usage = {
        "with_query_improvement": {
            "total_tokens": tokens_improved,
            "total_time": time_improved,
            "average_tokens_per_query": tokens_improved / len(questions),
            "average_time_per_query": time_improved / len(questions)
        },
        "without_query_improvement": {
            "total_tokens": tokens_original,
            "total_time": time_original,
            "average_tokens_per_query": tokens_original / len(questions),
            "average_time_per_query": time_original / len(questions)
        },
        "difference": {
            "total_tokens": tokens_improved - tokens_original,
            "total_time": time_improved - time_original,
            "average_tokens_per_query": (tokens_improved - tokens_original) / len(questions),
            "average_time_per_query": (time_improved - time_original) / len(questions)
        }
    }
    save_detailed_results(resource_usage, "overall_resource_usage_comparison.json")

    print("자원 사용 비교:")
    print(f"쿼리 개선 적용 시 - 총 토큰: {tokens_improved}, 총 시간: {time_improved:.2f}초")
    print(f"쿼리 개선 미적용 시 - 총 토큰: {tokens_original}, 총 시간: {time_original:.2f}초")
    print(f"차이 - 토큰: {tokens_improved - tokens_original}, 시간: {time_improved - time_original:.2f}초")
    print("각 질문별 상세 자원 사용량은 JSON 파일을 확인하세요.")
