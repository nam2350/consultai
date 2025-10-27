"""
센터링크 콜백 서버 (Centerlink Callback Server)

AI 배치 분석 결과를 수신하는 테스트용 콜백 서버입니다.
실제 연동 테스트 시 사용하며, 수신한 데이터를 콘솔과 파일에 저장합니다.

사용법:
    python centerlink_callback_server.py

외부 노출 (ngrok):
    ngrok http 5000

Author: AI 분석팀
Date: 2025-10-16
"""

from flask import Flask, request, jsonify
from datetime import datetime
import json
import os

app = Flask(__name__)

# 결과 저장 디렉토리
RESULTS_DIR = "callback_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.route('/api/ai-callback', methods=['POST'])
def receive_ai_callback():
    """
    AI 배치 분석 결과 수신 엔드포인트

    AI 서버에서 배치 처리 완료 후 이 엔드포인트로 결과를 전송합니다.
    """
    try:
        data = request.json
        timestamp = datetime.now()

        # 콘솔 출력
        print("\n" + "="*80)
        print(f"[AI 콜백 수신] {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"Batch ID      : {data.get('batch_id', 'N/A')}")
        print(f"Status        : {data.get('status', 'N/A')}")
        print(f"Total Count   : {data.get('total_count', 0)}")
        print(f"Success Count : {data.get('success_count', 0)}")
        print(f"Failed Count  : {data.get('failed_count', 0)}")
        print(f"Results       : {len(data.get('results', []))}개")

        # 각 통화별 결과 요약
        results = data.get('results', [])
        if results:
            print("\n[통화별 결과]")
            for idx, result in enumerate(results, 1):
                consultation_id = result.get('consultation_id', 'Unknown')
                success = result.get('success', False)
                processing_time = result.get('processing_time', 0)
                status_emoji = "✅" if success else "❌"

                print(f"  {idx}. {consultation_id}: {status_emoji} ({processing_time:.2f}초)")

        print("="*80)

        # 파일로 저장
        batch_id = data.get('batch_id', 'unknown')
        filename = f"{RESULTS_DIR}/batch_{batch_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[저장 완료] {filename}")
        print()

        # 반드시 200 OK 응답 (AI 서버가 재시도하지 않도록)
        return jsonify({
            "received": True,
            "batch_id": data.get('batch_id'),
            "timestamp": timestamp.isoformat(),
            "saved_to": filename
        }), 200

    except Exception as e:
        print(f"\n❌ [에러 발생] {e}")

        # 에러가 발생해도 200 OK 응답 (재시도 방지)
        return jsonify({
            "received": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({
        "status": "healthy",
        "service": "Centerlink Callback Server",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/', methods=['GET'])
def index():
    """루트 엔드포인트"""
    return jsonify({
        "service": "Centerlink Callback Server",
        "version": "1.0.0",
        "endpoints": {
            "callback": "/api/ai-callback (POST)",
            "health": "/health (GET)"
        },
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }), 200


if __name__ == '__main__':
    print("\n" + "="*80)
    print("센터링크 콜백 서버 시작")
    print("="*80)
    print(f"포트: 5000")
    print(f"콜백 엔드포인트: http://localhost:5000/api/ai-callback")
    print(f"헬스 체크: http://localhost:5000/health")
    print(f"결과 저장 폴더: {RESULTS_DIR}/")
    print("="*80)
    print("\n외부 노출을 위해 다른 터미널에서 실행:")
    print("  ngrok http 5000")
    print()

    app.run(host='0.0.0.0', port=5000, debug=False)
