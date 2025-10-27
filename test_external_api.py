"""
외부 API 테스트 스크립트

로컬 또는 외부 서버의 센터링크 연동 API를 테스트합니다.
"""
import requests
import json
import time

# 서버 URL 설정
BASE_URL = "http://localhost:8000"  # 로컬 테스트
# BASE_URL = "https://abc123xyz.ngrok-free.app"  # ngrok 사용 시
# BASE_URL = "http://192.168.0.100:8000"  # 같은 네트워크 내 다른 PC

# 바운드 키
BOUND_KEY = "test_key_centerlink_2025"

def test_health_check():
    """헬스 체크 테스트"""
    print("\n" + "="*80)
    print("[1] 헬스 체크 테스트")
    print("="*80)

    url = f"{BASE_URL}/api/v1/health"
    response = requests.get(url)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

    return response.status_code == 200

def test_realtime_api():
    """실시간 분석 API 테스트"""
    print("\n" + "="*80)
    print("[2] 실시간 분석 API 테스트 (SLM)")
    print("="*80)

    url = f"{BASE_URL}/api/v1/consultation/realtime-analyze"
    headers = {
        "Content-Type": "application/json",
        "X-Bound-Key": BOUND_KEY
    }
    data = {
        "bound_key": BOUND_KEY,
        "consultation_id": "EXTERNAL_TEST_REALTIME_001",
        "stt_data": {
            "conversation_text": (
                "상담사: 안녕하세요. 무엇을 도와드릴까요?\n"
                "고객: 보험 상품에 대해 문의드립니다.\n"
                "상담사: 어떤 보험 상품을 찾으시나요?\n"
                "고객: 건강 보험이요.\n"
                "상담사: 건강보험 상품을 안내해드리겠습니다."
            )
        }
    }

    print(f"요청 URL: {url}")
    print(f"요청 시작...")

    start_time = time.time()
    response = requests.post(url, headers=headers, json=data, timeout=30)
    elapsed = time.time() - start_time

    print(f"Status Code: {response.status_code}")
    print(f"처리 시간: {elapsed:.2f}초")
    print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")

    return response.status_code == 200

def test_batch_api():
    """배치 분석 API 테스트"""
    print("\n" + "="*80)
    print("[3] 배치 분석 API 테스트 (LLM)")
    print("="*80)

    url = f"{BASE_URL}/api/v1/consultation/batch-analyze-async"
    headers = {
        "Content-Type": "application/json",
        "X-Bound-Key": BOUND_KEY
    }
    data = {
        "bound_key": BOUND_KEY,
        "batch_id": "EXTERNAL_TEST_BATCH_001",
        "consultations": [
            {
                "consultation_id": "CALL_001",
                "stt_data": {
                    "conversation_text": (
                        "상담사: 안녕하세요.\n"
                        "고객: 대출 문의합니다.\n"
                        "상담사: 어떤 대출을 원하시나요?\n"
                        "고객: 주택담보대출이요."
                    )
                }
            },
            {
                "consultation_id": "CALL_002",
                "stt_data": {
                    "conversation_text": (
                        "상담사: 반갑습니다.\n"
                        "고객: 카드 발급 문의합니다.\n"
                        "상담사: 신용카드를 원하시나요?\n"
                        "고객: 네."
                    )
                }
            }
        ],
        "callback_url": "http://localhost:5000/api/ai-callback",
        "llm_model": "qwen3_4b",
        "priority": 1
    }

    print(f"요청 URL: {url}")
    print(f"배치 크기: {len(data['consultations'])}개")
    print(f"요청 시작...")

    response = requests.post(url, headers=headers, json=data, timeout=10)

    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")

    if response.status_code == 202:
        batch_id = response.json().get('batch_id')
        print(f"\n배치 접수 성공! Batch ID: {batch_id}")
        print(f"15-20초 후 콜백 URL로 결과가 전송됩니다.")

        # 배치 상태 조회
        print("\n배치 상태 조회 중...")
        time.sleep(2)

        status_url = f"{BASE_URL}/api/v1/consultation/batch-status/{batch_id}"
        status_response = requests.get(status_url, headers={"X-Bound-Key": BOUND_KEY})
        print(f"배치 상태:\n{json.dumps(status_response.json(), indent=2, ensure_ascii=False)}")

    return response.status_code == 202

def test_auth_failure():
    """인증 실패 테스트"""
    print("\n" + "="*80)
    print("[4] 인증 실패 테스트 (잘못된 키)")
    print("="*80)

    url = f"{BASE_URL}/api/v1/consultation/realtime-analyze"
    headers = {
        "Content-Type": "application/json",
        "X-Bound-Key": "invalid_key_12345"  # 잘못된 키
    }
    data = {
        "bound_key": "invalid_key_12345",
        "consultation_id": "AUTH_FAIL_TEST",
        "stt_data": {
            "conversation_text": "테스트"
        }
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)

    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print(f"예상대로 401 Unauthorized 발생: {'✅' if response.status_code == 401 else '❌'}")

    return response.status_code == 401

def main():
    """모든 테스트 실행"""
    print("\n" + "="*80)
    print("외부 API 테스트 시작")
    print(f"서버 URL: {BASE_URL}")
    print("="*80)

    results = []

    # 1. 헬스 체크
    try:
        results.append(("헬스 체크", test_health_check()))
    except Exception as e:
        print(f"❌ 헬스 체크 실패: {e}")
        results.append(("헬스 체크", False))
        return

    # 2. 실시간 API
    try:
        results.append(("실시간 API", test_realtime_api()))
    except Exception as e:
        print(f"❌ 실시간 API 실패: {e}")
        results.append(("실시간 API", False))

    # 3. 배치 API
    try:
        results.append(("배치 API", test_batch_api()))
    except Exception as e:
        print(f"❌ 배치 API 실패: {e}")
        results.append(("배치 API", False))

    # 4. 인증 실패
    try:
        results.append(("인증 실패", test_auth_failure()))
    except Exception as e:
        print(f"❌ 인증 실패 테스트 실패: {e}")
        results.append(("인증 실패", False))

    # 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)

    for test_name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{test_name:20s}: {status}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\n총 {total}개 테스트 중 {passed}개 성공 ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    main()
