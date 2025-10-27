#!/usr/bin/env python3
"""
캐싱 시스템 성능 테스트 스크립트

동일한 요청을 여러 번 보내서 캐시 성능 향상을 측정합니다.
"""

import asyncio
import time
import json
import statistics
from pathlib import Path
import httpx

class CachePerformanceTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def load_test_data(self):
        """테스트용 실제 통화 데이터 로드"""
        try:
            # call_data에서 실제 파일 하나 선택
            call_data_dir = Path("call_data/2025-07-15")
            json_files = list(call_data_dir.glob("*.json"))
            
            if not json_files:
                raise FileNotFoundError("테스트 데이터를 찾을 수 없습니다")
            
            # 첫 번째 파일 사용
            test_file = json_files[0]
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversation_text = data.get('conversation_text', '')
            if not conversation_text:
                raise ValueError("conversation_text가 없습니다")
            
            print(f"테스트 데이터: {test_file.name}")
            print(f"텍스트 길이: {len(conversation_text)} 문자")
            
            return {
                "consultation_id": f"cache_test_{int(time.time())}",
                "consultation_content": conversation_text,
                "stt_data": {
                    "conversation_text": conversation_text
                },
                "ai_tier": "llm",
                "llm_model": "qwen3_4b",
                "options": {
                    "include_summary": True,
                    "include_category_recommendation": True,
                    "include_title_generation": True,
                    "max_summary_length": 300
                }
            }
            
        except Exception as e:
            print(f"ERROR: 테스트 데이터 로드 실패: {e}")
            return None
    
    async def single_analysis_request(self, request_data):
        """단일 분석 요청 실행"""
        start_time = time.time()
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/consultation/analyze",
                json=request_data
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                processing_time = result.get('metadata', {}).get('processing_time', 0)
                model_used = result.get('metadata', {}).get('model_used', 'Unknown')
                cache_used = "(캐시)" in model_used
                
                return {
                    "success": True,
                    "total_time": duration,
                    "processing_time": processing_time,
                    "model_used": model_used,
                    "cache_used": cache_used,
                    "categories_count": len(result.get('results', {}).get('recommended_categories', [])),
                    "titles_count": len(result.get('results', {}).get('generated_titles', []))
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "total_time": duration
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "error": str(e),
                "total_time": end_time - start_time
            }
    
    async def run_cache_performance_test(self, iterations=5):
        """캐시 성능 테스트 실행"""
        print("캐시 성능 테스트 실행")
        print("=" * 50)
        
        # 테스트 데이터 로드
        test_data = await self.load_test_data()
        if not test_data:
            return
        
        results = []
        
        for i in range(iterations):
            print(f"\n[{i+1}/{iterations}] 분석 요청 실행...")
            
            result = await self.single_analysis_request(test_data)
            results.append(result)
            
            if result["success"]:
                cache_status = "CACHE HIT" if result["cache_used"] else "CACHE MISS"
                print(f"  상태: SUCCESS ({cache_status})")
                print(f"  총 시간: {result['total_time']:.2f}초")
                print(f"  처리 시간: {result['processing_time']:.2f}초")
                print(f"  모델: {result['model_used']}")
                print(f"  카테고리: {result['categories_count']}개")
                print(f"  제목: {result['titles_count']}개")
            else:
                print(f"  상태: FAILED - {result['error']}")
                print(f"  소요 시간: {result['total_time']:.2f}초")
        
        # 결과 분석
        await self.analyze_results(results)
    
    async def analyze_results(self, results):
        """테스트 결과 분석"""
        print("\n" + "=" * 50)
        print("📊 성능 테스트 결과 분석")
        print("=" * 50)
        
        successful_results = [r for r in results if r["success"]]
        if not successful_results:
            print("❌ 성공한 요청이 없습니다!")
            return
        
        # 캐시 히트/미스 분리
        cache_hits = [r for r in successful_results if r["cache_used"]]
        cache_misses = [r for r in successful_results if not r["cache_used"]]
        
        print(f"총 요청: {len(results)}개")
        print(f"성공: {len(successful_results)}개")
        print(f"캐시 히트: {len(cache_hits)}개")
        print(f"캐시 미스: {len(cache_misses)}개")
        
        if cache_misses:
            miss_times = [r["total_time"] for r in cache_misses]
            print(f"\n🔥 캐시 미스 (AI 분석):")
            print(f"  평균 시간: {statistics.mean(miss_times):.2f}초")
            print(f"  최소/최대: {min(miss_times):.2f}초 / {max(miss_times):.2f}초")
        
        if cache_hits:
            hit_times = [r["total_time"] for r in cache_hits]
            print(f"\n⚡ 캐시 히트 (즉시 반환):")
            print(f"  평균 시간: {statistics.mean(hit_times):.2f}초")
            print(f"  최소/최대: {min(hit_times):.2f}초 / {max(hit_times):.2f}초")
            
            if cache_misses:
                improvement = ((statistics.mean(miss_times) - statistics.mean(hit_times)) / statistics.mean(miss_times)) * 100
                speed_ratio = statistics.mean(miss_times) / statistics.mean(hit_times)
                print(f"\n🚀 성능 향상:")
                print(f"  개선률: {improvement:.1f}% 향상")
                print(f"  속도비: {speed_ratio:.1f}배 빨라짐")
        
        # 품질 일관성 확인
        if len(successful_results) > 1:
            categories_counts = [r["categories_count"] for r in successful_results]
            titles_counts = [r["titles_count"] for r in successful_results]
            
            categories_consistent = len(set(categories_counts)) == 1
            titles_consistent = len(set(titles_counts)) == 1
            
            print(f"\n✅ 품질 일관성:")
            print(f"  카테고리 개수 일치: {'YES' if categories_consistent else 'NO'}")
            print(f"  제목 개수 일치: {'YES' if titles_consistent else 'NO'}")
    
    async def close(self):
        """클라이언트 종료"""
        await self.client.aclose()

async def main():
    """메인 테스트 함수"""
    tester = CachePerformanceTester()
    
    try:
        await tester.run_cache_performance_test(iterations=3)
    finally:
        await tester.close()

if __name__ == "__main__":
    print("🧪 캐싱 시스템 성능 테스트")
    print("서버가 실행 중인지 확인하세요: python -m uvicorn main:app --reload")
    print()
    
    asyncio.run(main())
