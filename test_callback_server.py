"""
개발용 콜백 서버 (센터링크 시뮬레이션)

배치 분석 결과를 수신하는 센터링크 콜백 서버를 시뮬레이션합니다.
"""
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route('/api/ai-callback', methods=['POST'])
def receive_callback():
    """AI 분석 결과 콜백 수신"""
    data = request.json

    print("\n" + "="*80)
    print(f"[콜백 수신] {datetime.now()}")
    print("="*80)
    print(f"Batch ID: {data.get('batch_id')}")
    print(f"Status: {data.get('status')}")
    print(f"Total: {data.get('total_count')}")
    print(f"Success: {data.get('success_count')}")
    print(f"Failed: {data.get('failed_count')}")
    print(f"Processing Time: {data.get('total_processing_time')}초")
    print("\n[Results]:")

    for idx, result in enumerate(data.get('results', []), 1):
        print(f"\n{idx}. {result.get('consultation_id')}")
        print(f"   Success: {result.get('success')}")
        if result.get('success'):
            summary = result.get('summary', '')
            print(f"   Summary: {summary[:100]}...")
            print(f"   Categories: {result.get('categories', [])}")
            print(f"   Titles: {result.get('titles', [])}")
            print(f"   Model: {result.get('model')}")
            print(f"   Processing Time: {result.get('processing_time')}초")
        else:
            print(f"   Error: {result.get('error')}")

    print("="*80 + "\n")

    # 반드시 200 OK 응답
    return jsonify({
        "received": True,
        "batch_id": data.get('batch_id'),
        "timestamp": datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    print("\n" + "="*80)
    print("개발용 콜백 서버 시작")
    print("URL: http://localhost:5000/api/ai-callback")
    print("="*80 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
