"""파일/대화 처리 유틸리티: JSON I/O, STT 변환, 대화 텍스트 구성."""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def normalize_stt_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """call_data 전용 JSON 파싱 (초고속 최적화)"""
    segments = []
    
    try:
        # call_data 형식만 지원 (성능 극대화)
        details = data.get('raw_call_data', {}).get('details', [])
        if details:
            # 직접 매핑으로 최고 속도 달성
            segments = [
                {
                    'speaker': detail['senderType'],
                    'text': detail['message'].strip(),
                    'start': 0.0,
                    'end': 0.0
                }
                for detail in details
                if detail.get('senderType') and detail.get('message', '').strip()
            ]
        
    except Exception as e:
        logger.error(f"call_data JSON 파싱 실패: {e}")
        return []
    
    return segments

def validate_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """세그먼트 유효성 검증 및 정리"""
    validated = []
    
    for segment in segments:
        # 필수 필드 확인
        if not all(key in segment for key in ['speaker', 'text']):
            continue
            
        # 빈 텍스트 제외
        text = segment['text'].strip()
        # 과도한 필터링으로 대화문이 비는 문제를 완화하기 위해 1글자도 허용
        if not text or len(text) < 1:
            continue
            
        # 유효한 세그먼트 추가
        validated.append({
            'speaker': segment['speaker'],
            'text': text,
            'start': segment.get('start', 0.0),
            'end': segment.get('end', 0.0)
        })
    
    return validated

class FileProcessor:
    """파일 처리."""
    
    @staticmethod
    def load_json_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """JSON 로드."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"JSON 파일 로드 실패 {file_path}: {e}")
            return None
    
    @staticmethod
    def save_json_file(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """JSON 저장."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"JSON 파일 저장 실패 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_and_convert_stt_json(file_path_or_data: Union[str, Path, Dict, Any]) -> Optional[List[Dict]]:
        """STT JSON 로드/표준화."""
        try:
            # 파일 경로
            if isinstance(file_path_or_data, (str, Path)):
                data = FileProcessor.load_json_file(file_path_or_data)
                if not data:
                    return None
            # 데이터 객체
            else:
                data = file_path_or_data
            
            # STT JSON 변환/검증
            segments = normalize_stt_json(data)
            validated_segments = validate_segments(segments)
            
            if not validated_segments:
                logger.warning("변환된 세그먼트가 없습니다")
                return None
                
            return validated_segments
            
        except Exception as e:
            logger.error(f"STT JSON 변환 실패: {e}")
            return None

class ConversationProcessor:
    """대화 텍스트 처리."""
    
    @staticmethod
    def construct_conversation_text(segments: List[Dict]) -> str:
        """세그먼트 → 대화 텍스트 (call_data 전용 최적화)"""
        try:
            # call_data는 CUST/AGENT 포맷이므로 빠른 매핑 사용
            conversation_parts = []
            
            for segment in segments:
                speaker = segment.get('speaker', '')
                text = segment.get('text', '').strip()
                
                if not text:
                    continue
                    
                # call_data 전용 빠른 매핑
                if speaker == 'CUST':
                    conversation_parts.append(f"고객: {text}")
                elif speaker == 'AGENT':
                    conversation_parts.append(f"상담사: {text}")
                else:
                    # 기타 화자 (드문 케이스)
                    conversation_parts.append(f"{speaker}: {text}")
            
            return "\n".join(conversation_parts)
            
        except Exception as e:
            logger.error(f"대화 텍스트 구성 실패: {e}")
            return ""
    
    @staticmethod
    def extract_call_id(file_path: Union[str, Path]) -> str:
        """파일 경로 → 통화 ID."""
        filename = Path(file_path).stem
        # 확장자 제거 및 특수문자 정리
        call_id = filename.replace('.json', '').replace('.txt', '')
        return call_id

class ResultSaver:
    """결과 저장."""
    
    def __init__(self, output_base_dir: Optional[Union[str, Path]] = None):
        self.output_base_dir = Path(output_base_dir) if output_base_dir else Path("outputs")
    
    def save_summary_result(self, 
                          call_id: str,
                          model_name: str,
                          summary_text: str,
                          processing_time: float,
                          additional_data: Optional[Dict] = None) -> Dict[str, Any]:
        """요약 결과 저장."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = model_name.replace('/', '_').replace(' ', '_')
            
            # JSON 결과
            result_data = {
                'call_id': call_id,
                'model_name': model_name,
                'summary_text': summary_text,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            if additional_data:
                result_data.update(additional_data)
            
            # 파일 경로
            json_filename = f"{call_id}_{safe_model_name}_{timestamp}.json"
            txt_filename = f"{call_id}_{safe_model_name}_{timestamp}_summary.txt"
            
            json_filepath = self.output_base_dir / json_filename
            txt_filepath = self.output_base_dir / txt_filename
            
            # JSON 저장
            if FileProcessor.save_json_file(result_data, json_filepath):
                logger.info(f"JSON 결과 저장: {json_filepath}")
            
            # 텍스트 요약 저장
            try:
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"=== 상담 요약 결과 ===\n")
                    f.write(f"통화 ID: {call_id}\n")
                    f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"사용 모델: {model_name}\n")
                    f.write(f"처리 시간: {processing_time:.2f}초\n")
                    f.write("="*50 + "\n\n")
                    f.write("요약 내용:\n")
                    f.write(summary_text)
                
                logger.info(f"텍스트 요약 저장: {txt_filepath}")
                
            except Exception as e:
                logger.warning(f"텍스트 파일 저장 실패: {e}")
            
            return {
                'success': True,
                'json_file': str(json_filepath),
                'txt_file': str(txt_filepath),
                'call_id': call_id
            }
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'call_id': call_id
            }

# 전역 인스턴스
file_processor = FileProcessor()
conversation_processor = ConversationProcessor()
result_saver = ResultSaver()

# 편의 함수들
def extract_conversation_text(stt_data: Dict[str, Any]) -> Optional[str]:
    """
    STT 데이터 딕셔너리에서 대화 텍스트 추출 (센터링크 연동용)

    Args:
        stt_data: STT 데이터 딕셔너리
            - conversation_text: 대화 텍스트 (우선)
            - raw_call_data.details: 상세 데이터 (fallback)
            - segments: 세그먼트 배열 (fallback)

    Returns:
        대화 텍스트 또는 None
    """
    try:
        # 1. conversation_text 필드 우선 (가장 빠름)
        conversation_text = stt_data.get('conversation_text')
        if conversation_text and conversation_text.strip():
            return conversation_text.strip()

        # 2. raw_call_data.details에서 파싱 (call_data 형식)
        raw_call_data = stt_data.get('raw_call_data', {})
        details = raw_call_data.get('details', [])
        if details:
            segments = normalize_stt_json({'raw_call_data': raw_call_data})
            if segments:
                validated = validate_segments(segments)
                if validated:
                    return conversation_processor.construct_conversation_text(validated)

        # 3. segments 배열에서 직접 변환 (범용 형식)
        segments = stt_data.get('segments', [])
        if segments:
            validated = validate_segments(segments)
            if validated:
                return conversation_processor.construct_conversation_text(validated)

        logger.warning("STT 데이터에서 대화 텍스트를 추출할 수 없습니다")
        return None

    except Exception as e:
        logger.error(f"대화 텍스트 추출 실패: {e}")
        return None

def load_conversation_from_json(file_path: str) -> Optional[str]:
    """
    call_data JSON에서 대화 텍스트 로드 (초고속 최적화)

    Args:
        file_path: JSON 파일 경로

    Returns:
        대화 텍스트 또는 None
    """
    try:
        # JSON 파일 로드
        data = file_processor.load_json_file(file_path)
        if not data:
            return None

        # call_data는 conversation_text가 항상 준비되어 있음 (파싱 불필요!)
        conversation_text = data.get('conversation_text')
        if conversation_text and conversation_text.strip():
            return conversation_text.strip()

        # conversation_text가 없는 경우에만 fallback (극히 드문 경우)
        logger.warning(f"conversation_text가 없음, 파싱 시도: {file_path}")
        segments = file_processor.load_and_convert_stt_json(data)
        if segments:
            return conversation_processor.construct_conversation_text(segments)

        return None

    except Exception as e:
        logger.error(f"대화 텍스트 로드 실패 {file_path}: {e}")
        return None
