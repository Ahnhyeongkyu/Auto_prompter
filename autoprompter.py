from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import textwrap

# OpenAI API 키를 환경 변수에서 로드합니다.
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 사용할 모델을 지정합니다. OpenAI의 최신 모델을 사용합니다.
MODEL = "gpt-4o"  # 또는 "gpt-3.5-turbo"

def create_chat_completion(messages, max_tokens=1000):
    """
    OpenAI API를 사용하여 채팅 완성을 생성하는 함수입니다.
    
    :param messages: API에 전송할 메시지 목록
    :param max_tokens: 생성할 최대 토큰 수
    :return: 생성된 텍스트 응답
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
def chunk_text(text, chunk_size=2000):
    """긴 텍스트를 작은 청크로 나눕니다."""
    return textwrap.wrap(text, chunk_size, break_long_words=False, replace_whitespace=False)

def summarize_chunk(chunk):
    """텍스트 청크를 요약합니다."""
    messages = [
        {"role": "system", "content": "당신은 프롬프트 엔지니어링 전문가입니다. 주어진 텍스트를 간결하게 요약하되, 핵심 개념과 기법을 유지해야 합니다."},
        {"role": "user", "content": f"다음 프롬프트 엔지니어링 관련 텍스트를 요약해주세요:\n\n{chunk}"}
    ]
    return create_chat_completion(messages, max_tokens=500)

def extract_key_points(summary):
    """요약에서 핵심 포인트를 추출합니다."""
    messages = [
        {"role": "system", "content": "당신은 프롬프트 엔지니어링 전문가입니다. 주어진 요약에서 가장 중요한 핵심 포인트를 추출해야 합니다."},
        {"role": "user", "content": f"다음 요약에서 프롬프트 엔지니어링과 관련된 5-7개의 가장 중요한 핵심 포인트를 추출해주세요:\n\n{summary}"}
    ]
    return create_chat_completion(messages, max_tokens=300)

def combine_summaries(summaries):
    """여러 요약을 하나의 일관된 텍스트로 결합합니다."""
    combined = "\n\n".join(summaries)
    messages = [
        {"role": "system", "content": "당신은 프롬프트 엔지니어링 전문가입니다. 여러 요약을 하나의 일관되고 포괄적인 요약으로 통합해야 합니다."},
        {"role": "user", "content": f"다음 요약들을 하나의 일관되고 포괄적인 요약으로 통합해주세요:\n\n{combined}"}
    ]
    return create_chat_completion(messages, max_tokens=1000)

def summarize_content(content):
    """
    주어진 내용을 요약하는 개선된 함수입니다.
    
    :param content: 요약할 텍스트 내용
    :return: 요약된 텍스트와 핵심 포인트
    """
    chunks = chunk_text(content)
    chunk_summaries = [summarize_chunk(chunk) for chunk in chunks]
    combined_summary = combine_summaries(chunk_summaries)
    key_points = extract_key_points(combined_summary)
    
    final_summary = f"{combined_summary}\n\n핵심 포인트:\n{key_points}"
    return final_summary

def generate_system_prompt(summary):
    """
    요약된 내용을 바탕으로 시스템 프롬프트를 생성하는 함수입니다.
    
    :param summary: 요약된 텍스트
    :return: 생성된 시스템 프롬프트
    """
    messages = [
        {"role": "system", "content": "당신은 효과적인 시스템 프롬프트를 만드는 AI assistant입니다."},
        {"role": "user", "content": f"이 프롬프트 엔지니어링 기법 요약을 바탕으로, AI assistant가 다양한 작업을 더 잘 수행할 수 있도록 돕는 시스템 프롬프트를 만들어주세요:\n\n{summary}"}
    ]
    return create_chat_completion(messages)

def generate_diverse_prompt(summary, previous_prompts):
    """
    이전 프롬프트와 다른 새로운 시스템 프롬프트를 생성하는 함수입니다.
    
    :param summary: 요약된 텍스트
    :param previous_prompts: 이전에 생성된 프롬프트 목록
    :return: 새롭게 생성된 다양한 시스템 프롬프트
    """
    messages = [
        {"role": "system", "content": "당신은 효과적이고 다양한 시스템 프롬프트를 만드는 AI assistant입니다."},
        {"role": "user", "content": f"이 프롬프트 엔지니어링 기법 요약을 바탕으로, 다음 이전 프롬프트와는 다른 새롭고 독특한 시스템 프롬프트를 만들어주세요: {previous_prompts}\n\n요약:\n{summary}"}
    ]
    return create_chat_completion(messages)

def evolve_prompt(best_prompt, summary):
    """
    현재 최고의 프롬프트를 개선하는 함수입니다.
    
    :param best_prompt: 현재 최고의 프롬프트
    :param summary: 요약된 텍스트
    :return: 개선된 프롬프트
    """
    messages = [
        {"role": "system", "content": "당신은 시스템 프롬프트를 개선하는 데 특화된 AI assistant입니다. 현재 프롬프트를 분석하고, 개선된 버전의 프롬프트를 직접 제공해야 합니다."},
        {"role": "user", "content": f"이 시스템 프롬프트를 분석하고 프롬프트 엔지니어링 기법 요약을 바탕으로 개선된 버전의 프롬프트를 제공해주세요. 개선 사항에 대한 설명이 아닌, 실제로 사용할 수 있는 개선된 프롬프트 전체를 반환해야 합니다.\n\n현재 프롬프트: {best_prompt}\n\n요약: {summary}"}
    ]
    return create_chat_completion(messages)

def evaluate_prompt(prompt, task):
    """
    주어진 프롬프트와 태스크를 사용하여 AI의 응답을 생성하는 함수입니다.
    의료 QA 시스템에 맞게 수정됩니다.
    
    :param prompt: 시스템 프롬프트
    :param task: 수행할 태스크 (질문과 컨텍스트를 포함)
    :return: AI의 응답
    """
    question, context = task
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Question: {question}\nContext: {context}"}
    ]
    return create_chat_completion(messages)

def evaluate_response_quality(response, task):
    """
    AI 응답의 품질을 평가하는 함수입니다.
    의료 QA 시스템에 맞게 평가 기준을 수정합니다.
    
    :param response: AI의 응답
    :param task: 수행한 태스크 (질문과 컨텍스트를 포함)
    :return: 응답의 품질 점수 (1-10)
    """
    question, context = task
    messages = [
        {"role": "system", "content": "당신은 의료 QA 시스템의 응답을 평가하는 전문가입니다. 다음 기준으로 1-10점 척도로 평가하세요: 1) 정확성 2) 관련성 3) 이해 용이성 4) 완전성 5) 안전성"},
        {"role": "user", "content": f"다음 질문, 컨텍스트, 응답을 평가해주세요.\n질문: {question}\n컨텍스트: {context}\n응답: {response}\n평가는 'Rating: X/10' 형식으로 제공해주세요."}
    ]
    content = create_chat_completion(messages)
    
    try:
        rating = int(content.split('Rating:')[-1].strip().split('/')[0])
    except (ValueError, IndexError):
        rating = 5  # 기본값 설정
    
    return rating

def score_prompt(prompt, tasks):
    """
    여러 태스크에 대해 프롬프트의 성능을 평가하는 함수입니다.
    
    :param prompt: 평가할 시스템 프롬프트
    :param tasks: 테스트할 태스크 리스트
    :return: 프롬프트의 총점
    """
    total_score = 0
    for task in tasks:
        result = evaluate_prompt(prompt, task)
        score = evaluate_response_quality(result, task)
        total_score += score
    return total_score

def auto_prompter(summary, tasks, iterations=5, initial_prompt=None):
    """
    주어진 내용을 바탕으로 최적의 시스템 프롬프트를 찾는 함수입니다.
    
    :param content: 원본 텍스트 내용
    :param tasks: 테스트에 사용할 태스크 리스트
    :param iterations: 프롬프트 생성 및 평가를 반복할 횟수
    :param initial_prompt: 초기 시스템 프롬프트 (기본값: None)
    :return: 가장 높은 점수를 받은 시스템 프롬프트와 그 점수, 결과 리스트
    """
    if initial_prompt:
        best_prompt = initial_prompt
    else:
        best_prompt = generate_system_prompt(summary)
    best_score = score_prompt(best_prompt, tasks)
    previous_prompts = [best_prompt]
    
    results = []

    for i in range(iterations):
        print(f"반복 {i+1}/{iterations}")
        
        # 다양한 프롬프트 생성
        diverse_prompt = generate_diverse_prompt(summary, previous_prompts)
        diverse_score = score_prompt(diverse_prompt, tasks)
        
        # 최고의 프롬프트 진화
        evolved_prompt = evolve_prompt(best_prompt, summary)
        evolved_score = score_prompt(evolved_prompt, tasks)
        
        # 최고의 프롬프트 업데이트
        if diverse_score > best_score and diverse_score >= evolved_score:
            best_prompt = diverse_prompt
            best_score = diverse_score
            print(f"새로운 최고 프롬프트 (다양성). 점수: {best_score}")
        elif evolved_score > best_score:
            best_prompt = evolved_prompt
            best_score = evolved_score
            print(f"새로운 최고 프롬프트 (진화). 점수: {best_score}")
        else:
            print(f"개선 없음. 현재 최고 점수: {best_score}")
        
        previous_prompts.append(best_prompt)
        
        # 결과 저장
        results.append({
            "iteration": i+1,
            "best_prompt": best_prompt,
            "best_score": best_score
        })

    return best_prompt, best_score, results

def run_multiple_auto_prompter(content, tasks, iterations=5, num_runs=3, initial_prompt=None):
    """
    AutoPrompter를 여러 번 실행하고 최상의 결과를 반환하는 함수입니다.
    
    :param content: 원본 텍스트 내용
    :param tasks: 테스트에 사용할 태스크 리스트
    :param iterations: 각 실행에서의 반복 횟수
    :param num_runs: AutoPrompter를 실행할 총 횟수
    :return: 가장 높은 점수를 받은 시스템 프롬프트와 그 점수, 모든 실행의 결과
    """
    summary = summarize_content(content)  # 한 번만 요약
    best_overall_prompt = None
    best_overall_score = float('-inf')
    all_results = []

    for run in range(num_runs):
        print(f"\n실행 {run + 1}/{num_runs} 시작")
        best_prompt, best_score, results = auto_prompter(summary, tasks, iterations, initial_prompt)
        all_results.append({
            "run": run + 1,
            "best_prompt": best_prompt,
            "best_score": best_score,
            "iterations": results
        })

        if best_score > best_overall_score:
            best_overall_prompt = best_prompt
            best_overall_score = best_score
            print(f"새로운 최고 프롬프트 발견 (실행 {run + 1}). 점수: {best_overall_score}")
        else:
            print(f"실행 {run + 1} 완료. 최고 점수: {best_score}")

    return best_overall_prompt, best_overall_score, all_results

def save_multiple_run_results(results, filename=None):
    """
    여러 번의 실행 결과를 JSON 파일로 저장하는 함수입니다.
    
    :param results: 저장할 결과 데이터
    :param filename: 저장할 파일 이름 (기본값: None)
    """
    if filename is None:
        filename = f"auto_prompter_multiple_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 {filename}에 저장되었습니다.")

# 메인 실행 부분
if __name__ == "__main__":
    # mdx 파일을 읽어옵니다.
    with open("prompt_guide.mdx", "r", encoding="utf-8") as file:
        content = file.read()

     # 원래의 시스템 프롬프트
    original_prompt = """You are the world's most authoritative health checkup AI ChatGPT in the question-answering task in the field of medicine and healthcare, providing answers with given context to non-medical professionals. The given context is an excerpt of data in html format. If you answer accurately, you will be paid incentives 1million USD proportionally. Combine what you know and answer the questions in detail based on the given context. Please answer in a modified form so it looks nice."""

    # 테스트용 태스크 리스트를 정의합니다.
    tasks = [
        ("공복혈당장애와 내당능장애란 무엇인가요?", "<p>공복혈당장애는 공복 시 혈당이 정상보다 높지만 당뇨병 진단 기준에는 미치지 않는 상태를 말합니다. 내당능장애는 식후 혈당이 정상보다 높지만 당뇨병 진단 기준에는 미치지 않는 상태를 의미합니다.</p>"),
        ("단 것을 많이 먹으면 정말 내당능장애나 당뇨병이 생기나요?", "<p>과도한 당분 섭취는 내당능장애와 당뇨병 발생 위험을 높일 수 있습니다. 그러나 균형 잡힌 식단, 규칙적인 운동, 건강한 생활 습관을 유지하면 위험을 줄일 수 있습니다.</p>"),
        ("고혈압 약은 평생 먹어야 하나요?", "<p>고혈압은 만성 질환이므로 대부분의 경우 장기적인 약물 치료가 필요합니다. 그러나 생활 습관 개선으로 혈압이 정상화되면 의사와 상담 후 약물 용량을 조절하거나 중단할 수 있습니다.</p>"),
    ]

    # AutoPrompter를 여러 번 실행하여 최적의 시스템 프롬프트를 찾습니다.
    best_system_prompt, best_score, all_results = run_multiple_auto_prompter(content, tasks, iterations=5, num_runs=3, initial_prompt=original_prompt)
    
    # 결과를 출력합니다.
    print("\n최종 최고의 시스템 프롬프트:")
    print(best_system_prompt)
    print(f"\n최종 최고 점수: {best_score}")
    
    # 결과를 파일로 저장합니다.
    save_multiple_run_results({
        "best_overall_prompt": best_system_prompt,
        "best_overall_score": best_score,
        "all_runs": all_results
    })