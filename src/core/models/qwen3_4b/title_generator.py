"""
Qwen3-4B 기반 제목 생성기

상담 내용을 분석하여 키워드형과 설명형 제목을 생성합니다.
"""
import logging
import re
import torch
from typing import List, Dict, Any
from .summarizer import Qwen2507Summarizer

# 정규식 사전 컴파일 (성능 최적화)
UNDERSCORE_PATTERN = re.compile(r'_{2,}')
WHITESPACE_UNDERSCORE_PATTERN = re.compile(r'_\s+_')
SPACE_UNDERSCORE_PATTERN = re.compile(r'\s*_\s*')

logger = logging.getLogger(__name__)

class TitleGenerator:
    """Qwen3-4B를 사용한 제목 생성기"""
    
    def __init__(self, model_path: str = None, shared_summarizer: 'Qwen2507Summarizer' = None):
        """
        Args:
            model_path: Qwen 모델 경로
            shared_summarizer: 공유할 Qwen2507Summarizer 인스턴스 (성능 최적화용)
        """
        if shared_summarizer is not None:
            # 공유 summarizer 사용 (성능 최적화)
            self.summarizer = shared_summarizer
            self.model_loaded = True  # 이미 로드된 모델 사용
            self.use_shared_model = True  # 공유 모델 플래그
            
        else:
            # 독립 summarizer 생성 (기존 방식)
            self.summarizer = Qwen2507Summarizer(model_path)
            self.model_loaded = False
            self.use_shared_model = False
        
        # generation_params 제거 - summarizer의 파라미터 사용
    
    def load_model(self) -> bool:
        """모델 로드"""
        if not self.model_loaded:
            success = self.summarizer.load_model()
            self.model_loaded = success
            return success
        return True
    
    def generate(self, content: str) -> List[Dict[str, Any]]:
        """제목 생성 (키워드형/설명형)"""
        try:
            # 공유 모델 사용 시 로드 체크 완전 생략 (성능 최적화)
            if self.use_shared_model:
                pass
            elif not self.model_loaded:
                if not self.load_model():
                    raise RuntimeError("모델 로드 실패")
            
            # 범용적 제목 생성 프롬프트
            title_prompt = self._build_title_generation_prompt(content)

            # Qwen 모델을 사용하여 생성
            try:
                result = self._generate_response(title_prompt)
            except Exception:
                return []
            
            # 결과 파싱
            try:
                titles = self._parse_generated_titles(result)
            except Exception:
                titles = []
            # no-fallback + 상투어 필터 (강화)
            safe_titles = []
            try:
                for t in (titles or []):
                    if not isinstance(t, dict):
                        continue
                    _txt = t.get("title")
                    if not _txt or not str(_txt).strip():
                        continue
                    if t.get("source") == "fallback_message":
                        continue
                    s = str(_txt).lower()
                    # 의미없는 응답 필터링 강화
                    meaningless_patterns = [
                        "감사", "상담 완료", "상담 종료", "네", "예", "알겠",
                        "안녕", "죄송", "고맙", "thank", "yes", "ok", "okay",
                        "상담사:", "고객:", "응답:", "대화:"
                    ]
                    if any(phrase in s for phrase in meaningless_patterns):
                        continue
                    # 너무 짧은 제목 제외 (3자 미만)
                    if len(str(_txt).strip()) < 3:
                        continue
                    # 영어만 있는 제목 제외
                    if re.match(r'^[a-zA-Z\s\W]+$', str(_txt)):
                        continue
                    safe_titles.append(t)
            except Exception:
                safe_titles = []

            if not safe_titles:
                try:
                    rb = self._rule_based_title(content)
                except Exception:
                    rb = []
                safe_titles = rb

            return self._select_primary_title(safe_titles)
            
        except Exception as e:
            logger.error(f"[제목생성] 실패: {e}")
            # 가짜정보 생성 완전 금지 - 실패시 안내 메시지 반환
            return [{
                "title": "제목 생성을 못했습니다.",
                "type": "descriptive",
                "confidence": 0.0,
                "source": "fallback_message"
            }]
    
    def _generate_response(self, prompt: str) -> str:
        """모델 응답 생성 - WhisperX_web 최적화된 방식"""
        try:
            # 요약기의 최적화된 생성 방식 직접 사용
            model_inputs = self.summarizer.tokenizer([prompt], return_tensors="pt").to(self.summarizer.device)
            
            with torch.no_grad():
                generated_ids = self.summarizer.model.generate(
                    model_inputs['input_ids'],
                    attention_mask=model_inputs['attention_mask'],
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.7,
                    top_k=20,
                    top_p=0.8,
                    min_p=0.0,
                    repetition_penalty=1.05,
                    early_stopping=True,
                    pad_token_id=self.summarizer.tokenizer.pad_token_id,
                    eos_token_id=self.summarizer.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # 디코딩 
            response_ids = generated_ids[0][len(model_inputs['input_ids'][0]):]
            response = self.summarizer.tokenizer.decode(response_ids, skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            logger.error(f"[응답생성] 실패: {e}")
            raise
    
    def _build_title_generation_prompt(self, content: str) -> str:
        """범용적 제목 생성 프롬프트"""
        system_prompt = (
            "당신은 상담 대화에서 핵심 제목을 작성하는 전문가입니다.\n"
            "다음 상담 내용을 모두 읽고, 대화에 근거한 제목 1개를 작성하세요.\n"
            "규칙:\n"
            "1. 대화에서 언급된 구체적인 주제나 문제를 중심으로 작성\n"
            "2. 감사 인사, 인사말, 단순 응답은 제목으로 사용 금지\n"
            "3. 특정 업종이나 도메인 용어에 편향되지 않게 작성\n"
            "4. 30자 이내로 간결하게 작성\n\n"
            "형식: 상담 제목: [핵심 내용 요약]\n\n"
            "예시:\n"
            "- 상담 제목: 서비스 이용 중 오류 발생 문의\n"
            "- 상담 제목: 계정 비밀번호 재설정 요청\n"
            "- 상담 제목: 제품 배송 지연 관련 확인\n"
        )
        
        return f"{system_prompt}\n\n상담내용:\n{content}"
    
    def _parse_generated_titles(self, raw_result: str) -> List[Dict[str, Any]]:
        """범용적 제목 파싱 - 하드코딩 제거"""
        try:
            # 프롬프트 누출 감지: Human:이 줄 시작부분에 나타나는 경우만 차단
            lines_raw = raw_result.strip().split('\n')
            if any(line.strip().startswith("Human:") or line.strip().startswith("당신은") for line in lines_raw):
                logger.warning("[제목생성] 프롬프트 누출 감지, 빈 결과 반환")
                return []
            
            titles = []
            lines = raw_result.strip().split('\n')
            
            for line in lines:
                # Human: 텍스트 중간에 나타나는 경우 제거
                if "Human:" in line:
                    line = line.split("Human:")[0]
                # 영어 텍스트 패턴 제거 (강화)
                if any(eng_word in line for eng_word in ["Humanity", "is a great", "We'd", "love to", "would", "could"]):
                    continue
                # 의미불명 패턴 제거
                if "(" in line and ")" in line and len(line.split()) <= 8:
                    continue
                line = line.strip()
                if not line:
                    continue
                    
                # 단순화된 제목 파싱 - "상담 제목:" 형식
                if "상담 제목:" in line:
                    title = line.split("상담 제목:", 1)[-1].strip()
                    # 길이 제한 및 유효성 체크 (30자 이내)
                    if title and len(title) <= 50:  # 여유분 포함
                        titles.append({
                            "title": title,
                            "type": "descriptive",
                            "confidence": 0.90,
                            "source": "ai_generated"
                        })
                # 혹시 형식 없이 바로 제목만 나오는 경우도 처리
                elif len(line) <= 50 and len(line) >= 5:  # 적절한 길이의 제목
                    titles.append({
                        "title": line,
                        "type": "descriptive", 
                        "confidence": 0.85,
                        "source": "ai_generated"
                    })
            
            # 중복 제거 - 동일한 제목과 타입의 중복 항목 제거
            seen_titles = set()
            unique_titles = []
            for title_obj in titles:
                title_key = (title_obj['title'], title_obj['type'])
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_titles.append(title_obj)
            
            # 최대 1개만 반환 (단순화된 제목)
            if unique_titles:
                return unique_titles[:1]
            else:
                # 제목 생성 실패 시 안내 메시지 반환
                return [{
                    "title": "제목 생성을 못했습니다.",
                    "type": "descriptive",
                    "confidence": 0.0,
                    "source": "fallback_message"
                }]
            
        except Exception as e:
            logger.error(f"[제목파싱] 실패: {e}")
            # 가짜정보 생성 완전 금지 - 실패시 안내 메시지 반환
            return [{
                "title": "제목 생성을 못했습니다.",
                "type": "descriptive", 
                "confidence": 0.0,
                "source": "fallback_message"
            }]
    
    def _select_primary_title(self, titles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pick a single best title entry for downstream consumers"""
        if not titles:
            return []
        best = None
        best_score = float('-inf')
        for candidate in titles:
            if not isinstance(candidate, dict):
                continue
            score = candidate.get('confidence', 0.0)
            if best is None or score > best_score:
                best = candidate
                best_score = score
        if best is None:
            best = next((t for t in titles if isinstance(t, dict)), None)
        return [best] if best else []

    def _rule_based_title(self, content: str) -> List[Dict[str, Any]]:
        """도메인 비종속 규칙형 제목 생성: 본문에서 동적 키워드/행위 추출.
        - 고정 사전 최소화, 불용어/부자연 표현 제거
        - [핵심명사] [행위] 어순 우선, 메뉴경로 존재 시 선반영
        """
        try:
            if not content or not isinstance(content, str):
                return []
            text = content
            # 1) 메뉴 경로 패턴(>, /, ->) 우선 추출
            menu_path = None
            for pat in [r"([가-힣A-Za-z0-9\s]+>(?:\s*[가-힣A-Za-z0-9]+)+)",
                        r"([가-힣A-Za-z0-9\s]+/(?:\s*[가-힣A-Za-z0-9]+)+)",
                        r"([가-힣A-Za-z0-9\s]+->(?:\s*[가-힣A-Za-z0-9]+)+)"]:
                m = re.search(pat, text)
                if m:
                    menu_path = re.sub(r"\s+", " ", m.group(1)).strip()
                    break
            # 2) 동적 핵심 명사 후보 추출(간단 빈도 기반 n-gram, 불용어 제외)
            # 한글/영문/숫자/공백만 유지
            cleaned = re.sub(r"[^가-힣A-Za-z0-9\s>/-]", " ", text)
            tokens = [t for t in re.split(r"\s+", cleaned) if t]
            stop = set([
                "고객","상담사","상담","결과","문의","안내","그리고","또는","합니다","했습니다","있습니다","하는","하여",
                "및","등","으로","에서","에게","께","에게서","으로부터","부터","까지","은","는","이","가","을","를","에","의",
                "이다","또","더","됨","관련","사항","내용","예","예시","경우","사용","처리","진행",
                "어떻게","되실까요","라는","이면","이면","같은","부분","기준","사항입니다","입니다","해당","추가",
                "감사","감사합니다","고맙습니다","죄송","죄송합니다","안녕","안녕하세요","네","예","알겠습니다"
            ])
            invalid_substrings = ["어떻게","되실까요","라는","까지","으로","으로써","이며","이고","이고요","랍니다","겠어요"]
            # 단어길이/숫자 비중 필터
            def is_candidate_word(w:str)->bool:
                if len(w) < 2 or len(w) > 12:
                    return False
                if w.isdigit():
                    return False
                if any(ch.isdigit() for ch in w):
                    # 연/숫자 포함 단어는 기본 제외(도메인 독립성/환각 방지)
                    return False
                if w in stop:
                    return False
                if any(bad in w for bad in invalid_substrings):
                    return False
                return True
            # 빈도 계산
            from collections import Counter
            c1 = Counter([w for w in tokens if is_candidate_word(w)])
            # 2-gram 결합 시도(간단히 인접 단어 결합)
            bigrams = []
            for i in range(len(tokens)-1):
                a,b = tokens[i], tokens[i+1]
                if is_candidate_word(a) and is_candidate_word(b):
                    if any(bad in a for bad in invalid_substrings) or any(bad in b for bad in invalid_substrings):
                        continue
                    bigrams.append(f"{a} {b}")
            c2 = Counter(bigrams)
            # 상위 후보 선택(2-gram 우선, 없으면 1-gram)
            dom = None
            if c2:
                dom_candidate = c2.most_common(1)[0][0]
                if not any(bad in dom_candidate for bad in invalid_substrings):
                    dom = dom_candidate
            elif c1:
                dom = c1.most_common(1)[0][0]
            # 3) 행위어는 범용 세트에서 동적 탐지(도메인 무관 공통 동사/명사)
            actions = [
                "취소","등록","정정","변경","조회","신청","승인","반려","정리","확인","발급","연결","안내",
                "해제","배정","이관","삭제","복구","확인취소","철회","반환","추가","수정","갱신"
            ]
            act = next((w for w in actions if w in text), None)
            # 4) 제목 후보 조립(메뉴경로가 있으면 우선 포함)
            candidates: List[str] = []
            if menu_path and act and dom:
                candidates.append(f"{menu_path} {act} – {dom}")
            if dom and act and not candidates:
                candidates.append(f"{dom} {act} 안내")
            if dom and not candidates:
                if "문의" in text:
                    candidates.append(f"{dom} 문의")
                else:
                    candidates.append(f"{dom} 안내")
            if not candidates and act:
                candidates.append(f"{act} 방법")
            if menu_path and not candidates:
                candidates.append(f"{menu_path} 안내")
            # 5) 정규화 및 제한
            def normalize_title(s: str) -> str:
                s = s.strip()
                for bad in ["감사","상담 완료","상담 종료"]:
                    s = s.replace(bad, "").strip()
                # 과도한 공백/기호 정리
                s = re.sub(r"\s+", " ", s)
                # 어색한 조사/부사 제거
                for tail in [" 라는"," 라는 것"," 까지"," 으로"," 으로써"," 으로의"," 으로부터"," 에 대한"," 에 관한"]:
                    if s.endswith(tail):
                        s = s[: -len(tail)].strip()
                # 문장 내 어색한 구절 정리(서술형 어미/조사)
                s = re.sub(r"\b(합니다|합니다\.|합니다요|입니다|입니다\.)\b", "", s).strip()
                s = re.sub(r"\s+라는\s+", " ", s)
                s = re.sub(r"\s+까지\b", "", s)
                s = re.sub(r"\s+으로\b", "", s)
                # 정중/요청 표현 제거
                polite = [
                    "말씀해 주시겠습니까","주시겠습니까","주시기 바랍니다","가능할까요","해주세요","해 주세요",
                    "부탁드립니다","문의드립니다","문의 드립니다","요청드립니다","요청 드립니다"
                ]
                for p in polite:
                    s = s.replace(p, "").strip()
                # 길이 제한
                if len(s) > 35:
                    s = s[:35]
                # 이중 공백 정리 재적용
                s = re.sub(r"\s+", " ", s).strip()
                return s
            candidates = [normalize_title(c) for c in candidates if c and c.strip()]
            # 한글 우선: 한글 미포함 후보 제거
            candidates = [c for c in candidates if re.search(r"[가-힣]", c)]
            candidates = [c for c in candidates if 5 <= len(c) <= 35]
            # 지나치게 일반적인 후보 제거(단일 동사/명사 + 안내/방법만)
            def too_generic(title: str) -> bool:
                toks = [t for t in re.split(r"\s+", title) if t]
                if len(toks) < 2:
                    return True
                if len(toks) == 2 and toks[0] in ["확인","진행","처리"] and toks[1] in ["안내","방법"]:
                    return True
                return False
            candidates = [c for c in candidates if not too_generic(c)]
            if not candidates:
                return []
            title = candidates[0]
            return [{
                "title": title,
                "type": "descriptive",
                "confidence": 0.9,
                "source": "keyword_rule"
            }]
        except Exception:
            return []

    def _clean_underscores(self, text: str) -> str:
        """키워드형 제목의 언더바 정리 (최적화됨)"""
        if not text:
            return text
        
        # 사전 컴파일된 정규식 사용 (성능 향상)
        cleaned = UNDERSCORE_PATTERN.sub('_', text)
        cleaned = cleaned.strip('_')
        cleaned = WHITESPACE_UNDERSCORE_PATTERN.sub('_', cleaned)
        cleaned = SPACE_UNDERSCORE_PATTERN.sub('_', cleaned)
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def cleanup(self):
        """리소스 정리 (공유 모델은 정리하지 않음)"""
        # 공유 summarizer를 사용하는 경우 cleanup하지 않음  
        if hasattr(self, 'summarizer') and self.summarizer and not hasattr(self.summarizer, '_is_shared'):
            self.summarizer.cleanup()
        self.model_loaded = False
