"""
CareerTube Backend -- Flask API v3.4
유튜브 시청 기록 기반 진로 추천 서비스

변경사항 (v3.4):
  - 유튜브 플랫폼 단어(영상/채널/유튜브/youtube) STOPWORDS 완전 제거
  - 엔터테인먼트 노이즈 단어 대폭 확장 (예능/레전드/먹방 등)
  - 진로 신호 단어 부스트: _CAREER_SIGNAL_BOOST (2.5×)
  - 구독 채널명 → 진로 키워드 자동 주입: _CHANNEL_CAREER_MAPPING
  - '유튜버' 직업 제거, '음악가·뮤지션' 신규 추가
  - 채널 가중치 6×, 키워드 텍스트 5× 강화

실행:
  pip install flask flask-cors scikit-learn requests
  python app.py
"""

import os
import re
import json
import requests
from math import log
from collections import defaultdict
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 앱 초기화
# ============================================================
app = Flask(__name__)
CORS(app)

GOOGLE_SHEET_URL = os.getenv(
    "GOOGLE_SHEET_URL",
    "https://script.google.com/macros/s/AKfycbwnFE-GU9d7HwVQO4huuKhorCqSOhPdXsU7HZEOF2ASQwd7MrtYV9S-L5Xs9xAbGAFQYw/exec"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_DB_PATH = os.path.join(BASE_DIR, "job_db.json")


# ============================================================
# 스마트 키워드 추출
# ============================================================

_PARTICLE_RE = re.compile(
    r"(을|를|이|가|은|는|의|에서|에게|한테|에|로|으로|와|과|도|만|"
    r"부터|까지|보다|처럼|같이|으로서|로서|이라|라고|고서|하고|"
    r"서|면|며|지만|하지만|이고|이며|이나|이든|이면)$"
)

_STOPWORDS = {
    # ── 기본 한국어 불용어 ─────────────────────────────────────
    "것", "수", "때", "중", "전", "후", "등", "및", "약", "좀", "잘", "더",
    "이", "가", "을", "를", "은", "는", "에", "의", "와", "과", "도", "만",
    "그", "저", "이거", "저거", "그거", "여기", "저기", "거기",
    "뭔가", "어떤", "무엇", "왜", "언제", "어디", "누가", "무슨",
    "이런", "그런", "저런", "모든", "아주", "너무", "정말", "진짜",
    "완전", "매우", "더욱", "이미", "아직", "다시", "바로", "제일",
    "어떻게", "이렇게", "그렇게", "저렇게", "같이", "함께", "혼자",
    "우리", "나는", "제가", "저의", "내가", "여러", "각각", "또한",
    "역시", "대로", "위해", "위한", "대한", "대해", "통해", "통한",
    "관련", "관한", "해서", "하면서", "하는", "하고", "있는", "없는",
    "된", "되는", "될", "하지만", "그리고", "그러나", "그래서",
    "그런데", "한번", "있어", "없어", "해요", "했어", "해야", "하기",
    "되기", "알아보", "알아보기", "알아볼", "알아보는", "먹는", "보는",
    "그냥", "약간", "살짝", "제대로", "드디어", "결국", "마침내",
    "처음", "마지막", "분", "명", "개", "번", "회", "권", "편", "개월",
    # ── 기본 영어 불용어 ──────────────────────────────────────
    "the", "a", "an", "in", "on", "at", "of", "for", "to", "is", "are",
    "was", "were", "be", "been", "have", "has", "do", "does", "did",
    "with", "by", "from", "as", "this", "that", "it", "its", "and", "or",
    "but", "not", "no", "can", "will", "my", "your", "we", "they", "i",
    "how", "what", "when", "why", "who", "which", "up", "out", "about",
    "if", "than", "so", "just", "get", "got", "go", "going", "vs",
    "ep", "ft", "mr", "dr", "etc", "im", "me", "he", "she", "its",
    # ── YouTube / SNS 플랫폼 단어 (진로 신호 아님) ────────────
    "youtube", "youtu", "www", "http", "https", "com", "co", "kr", "net",
    "watch", "video", "shorts", "channel", "playlist", "subscribe",
    "official", "tv", "clip", "highlight", "하이라이트", "채널",
    "구독", "구독자", "좋아요", "알림", "공식", "영상", "동영상", "유튜브",
    "재생목록", "쇼츠", "릴스", "reels", "tiktok", "틱톡",
    "조회수", "댓글", "썸네일", "업로드", "인플루언서", "vlog", "브이로그",
    "sns", "instagram", "인스타", "인스타그램", "페이스북", "트위터",
    # ── 예능/엔터테인먼트 노이즈 (직업 신호 아님) ─────────────
    "레전드", "레전", "웃참", "참교육", "근황", "결말", "포함", "스포",
    "먹방", "언박싱", "리액션", "reaction",
    "갑자기", "역사상", "역대", "최초", "최고", "최강",
    "충격", "대박", "실화", "소름", "충격적",
    "예능", "개그", "코미디",
    # ── 제목 클릭베이트 패턴 ─────────────────────────────────
    "이유", "방법", "순간", "상황", "사실", "진실", "비밀", "진심",
    "몰랐던", "모르는", "알려주는", "알아야", "놀라운", "대단한",
    "라이브", "live", "official", "mv",
    # ── 유닛/숫자 불용어 ─────────────────────────────────────
    "top", "part", "ep", "no", "vol",
    # ── 외모/생활 스타일 노이즈 ──────────────────────────────
    "장발", "단발", "남자", "여자", "남성", "여성", "외모", "패션",
    # ── 엔터테인먼트 특정 채널/그룹명 노이즈 ─────────────────
    "동네놈들", "hoodboyz", "리센느", "rescene",
    # ── 게임 관련 노이즈 (진로 신호 아님) ────────────────────
    "전직", "update", "업데이트", "패치", "게임", "플레이어",
    # ── 일반 잡음 단어 ────────────────────────────────────────
    "세상", "가장", "생각", "기다리", "요즘", "반응", "모음",
    "인생", "feat", "오지", "선수",
}

# 진로 관련 신호 단어: 이 단어들이 포함된 키워드는 점수 2.5배 부스트
# → 빈도가 낮아도 진로 의미가 있는 단어가 상위에 올라오도록
_CAREER_SIGNAL_BOOST = {
    # 음악·퍼포밍아츠
    "드럼", "drum", "트럼펫", "trumpet", "피아노", "기타", "바이올린",
    "재즈", "jazz", "음악", "music", "연주", "악기", "밴드", "band",
    "작곡", "편곡", "화성", "리듬", "앙상블", "공연",
    # IT·개발·데이터
    "코딩", "coding", "개발", "developer", "파이썬", "python",
    "프로그래밍", "programming", "알고리즘", "데이터", "data",
    "인공지능", "머신러닝", "딥러닝", "ai", "backend", "frontend",
    # 수학·과학·탐구
    "수학", "통계", "과학", "물리", "화학", "생물", "탐구", "사고실험",
    "논리", "증명", "방정식", "미적분",
    # 자기계발·성장
    "자기계발", "자기개발", "동기부여", "성장", "독서", "루틴", "습관",
    "생산성", "목표", "성공", "도전",
    # 비즈니스·창업
    "창업", "스타트업", "마케팅", "비즈니스", "투자", "재테크",
    "경영", "전략", "브랜딩", "스케일업",
    # 교육·학습
    "공부", "학습", "입시", "강의", "교육", "영어", "토익", "어학",
    # 심리·상담
    "심리", "상담", "힐링", "뇌과학", "정신건강", "감정",
    # 운동·건강
    "운동", "헬스", "피트니스", "트레이닝", "체력",
}

# 해시태그로 표시돼도 의미 있는 주제 단어 허용 목록
# → #리센느 #도아 같은 고유명사는 제거, #커플 #드럼 같은 주제어는 유지
_TOPIC_HASHTAGS = {
    "커플", "브이로그", "여행", "먹방", "일상", "운동", "공부", "독서",
    "음악", "드럼", "피아노", "기타", "바이올린", "트럼펫", "첼로",
    "디자인", "사진", "그림", "미술", "요리", "카페", "게임",
    "영화", "드라마", "독학", "성장", "동기부여", "습관", "루틴",
    "생산성", "재테크", "투자", "창업", "스타트업", "직장", "취업",
    "강의", "스터디", "수능", "입시", "편입", "자격증",
    "헬스", "다이어트", "필라테스", "러닝", "등산", "수영",
    "패션", "뷰티", "메이크업", "스킨케어", "인테리어", "살림",
    "반려동물", "강아지", "고양이", "심리", "mbti", "힐링",
    "코딩", "개발", "ai", "데이터", "마케팅", "광고", "브랜딩",
    "스포츠", "축구", "농구", "야구", "테니스",
}


def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    token = _PARTICLE_RE.sub("", token)
    return token


def _tokenize_title(title: str) -> list:
    """
    제목을 의미 단위 토큰으로 분해.
    - 해시태그(#단어): 주제 허용 목록에 있는 것만 유지, 나머지 고유명사 제거
    - 특수문자 제거 후 공백 기준 분리
    - 조사 제거 + 불용어 필터 + 한국어 자음/모음 노이즈 제거
    """
    # 해시태그 처리: 주제 허용 목록에 있으면 단어만 남기고, 아니면 제거
    def _replace_hashtag(m):
        word = m.group(1)
        norm = _normalize_token(word)
        return norm if norm in _TOPIC_HASHTAGS else " "
    title = re.sub(r"#(\S+)", _replace_hashtag, title)

    cleaned = re.sub(r"[^\w\s가-힣]", " ", title)
    tokens = []
    for raw in cleaned.split():
        norm = _normalize_token(raw)
        if re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣ]+", norm):
            continue
        if len(norm) >= 2 and norm not in _STOPWORDS:
            tokens.append(norm)
    return tokens


# 구독 채널명 → 진로 키워드 자동 매핑
# 채널 이름에 트리거 단어가 있으면 해당 진로 키워드를 keyword 목록에 주입
_CHANNEL_CAREER_MAPPING = [
    (["수학", "math", "깨봉"],              ["수학", "인공지능수학"]),
    (["인공지능", "ai", "artificial"],       ["인공지능", "ai"]),
    (["사고실험"],                           ["사고실험", "과학적사고"]),
    (["탐구생활", "탐구"],                   ["탐구", "지식탐구"]),
    (["창업", "스타트업", "entrepreneur"],   ["창업", "스타트업"]),
    (["자기계발", "동기부여", "작심", "라이프코드", "터닝포인트", "성공"],
                                             ["자기계발", "동기부여", "성장"]),
    (["심리", "정신과", "상담"],             ["심리", "심리상담"]),
    (["입시", "수능"],                       ["입시", "교육"]),
    (["재테크", "투자", "금융", "금닥터"],   ["재테크", "투자"]),
    (["독서", "book", "책"],                 ["독서", "자기개발"]),
    (["코딩", "개발", "python", "it"],       ["코딩", "개발"]),
    (["드럼", "drum"],                       ["드럼", "음악"]),
    (["트럼펫", "trumpet"],                  ["트럼펫", "음악"]),
    (["재즈", "jazz"],                       ["재즈", "음악"]),
    (["피아노", "piano"],                    ["피아노", "음악"]),
    (["운동", "헬스", "fitness"],            ["운동", "헬스"]),
    (["글쓰기", "작가", "writer"],           ["글쓰기", "창작"]),
]


def inject_channel_keywords(channels: list) -> list:
    """
    구독 채널명을 분석해 진로 관련 키워드를 추출·반환.
    시청 기록 빈도와 무관하게 구독 자체가 관심사를 나타내므로
    최종 keyword 목록에 합산해 진로 신호 강도를 높임.
    """
    injected = []
    for ch in channels:
        ch_lower = ch.lower()
        for triggers, keywords in _CHANNEL_CAREER_MAPPING:
            if any(t.lower() in ch_lower for t in triggers):
                for kw in keywords:
                    if kw not in injected:
                        injected.append(kw)
    return injected


def extract_keywords_smart(titles: list, top_n: int = 60) -> list:
    """
    Interest Intensity Score 기반 스마트 키워드 추출.
    Score = TF × log(1+DF) × (1+Coverage)
    Bigram에 1.5× 가중치 적용.
    """
    if not titles:
        return []

    num_titles = len(titles)
    tokenized = [_tokenize_title(t) for t in titles]

    term_tf: dict = defaultdict(int)
    term_df: dict = defaultdict(int)
    for tokens in tokenized:
        seen = set()
        for tok in tokens:
            term_tf[tok] += 1
            if tok not in seen:
                term_df[tok] += 1
                seen.add(tok)

    unigram_scores: dict = {}
    for term, tf in term_tf.items():
        df = term_df[term]
        if df < 3:   # 3개 미만 제목에서만 나오면 노이즈일 가능성 높음
            continue
        coverage = df / num_titles
        base_score = tf * log(1 + df) * (1 + coverage)
        # 진로 신호 단어는 2.5배 부스트 → 빈도 낮아도 표면에 올라오도록
        boost = 2.5 if term in _CAREER_SIGNAL_BOOST else 1.0
        unigram_scores[term] = base_score * boost

    bigram_tf: dict = defaultdict(int)
    bigram_df: dict = defaultdict(int)
    for tokens in tokenized:
        seen_bg = set()
        for i in range(len(tokens) - 1):
            bg = tokens[i] + " " + tokens[i + 1]
            bigram_tf[bg] += 1
            if bg not in seen_bg:
                bigram_df[bg] += 1
                seen_bg.add(bg)

    bigram_scores: dict = {}
    for bg, tf in bigram_tf.items():
        df = bigram_df[bg]
        if df < 3:
            continue
        coverage = df / num_titles
        base_score = tf * log(1 + df) * (1 + coverage) * 1.5
        # bigram 구성 토큰 중 하나라도 진로 신호어면 부스트
        bg_tokens = bg.split()
        boost = 2.0 if any(tok in _CAREER_SIGNAL_BOOST for tok in bg_tokens) else 1.0
        bigram_scores[bg] = base_score * boost

    top_bigrams = sorted(bigram_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    bigram_component_tokens = set()
    for bg, _ in top_bigrams:
        for tok in bg.split():
            bigram_component_tokens.add(tok)

    filtered_unigrams = {
        k: v for k, v in unigram_scores.items()
        if k not in bigram_component_tokens
    }
    remaining = top_n - len(top_bigrams)
    top_unigrams = sorted(filtered_unigrams.items(), key=lambda x: x[1], reverse=True)[:remaining]

    final_keywords = (
        [bg for bg, _ in top_bigrams]
        + [tok for tok, _ in top_unigrams]
    )
    return final_keywords[:top_n]


# ============================================================
# 콘텐츠 클러스터 & 심리적 동기 (SDT 기반)
# ============================================================
CONTENT_CLUSTERS = {
    "entertainment": {
        "label": "엔터테인먼트",
        "emoji": "🎭",
        "seeds": [
            "예능", "브이로그", "vlog", "먹방", "리액션", "일상", "웃긴", "개그", "코미디",
            "드라마", "아이돌", "kpop", "연예", "배우", "영화", "게임방송",
            "쇼츠", "챌린지", "밈", "힐링", "funny",
        ],
        "motivation": {
            "core": "관계·소속 욕구",
            "detail": "콘텐츠를 통해 간접적으로 관계를 경험하고 감정을 나누려는 욕구가 강합니다.",
            "sdt": "관계성 (Relatedness)",
            "career_hint": "사람과 소통하고 감정을 전달하는 직무에 강점이 있을 수 있습니다.",
        },
    },
    "self_development": {
        "label": "자기개발",
        "emoji": "📈",
        "seeds": [
            "공부", "루틴", "생산성", "동기부여", "성장", "습관", "독서", "자기계발", "마인드셋",
            "목표", "시간관리", "집중력", "독학", "강의", "도전", "성공", "아침루틴",
            "효율", "자기혁신", "변화",
        ],
        "motivation": {
            "core": "성장·자기효능감 욕구",
            "detail": "더 나은 자신이 되고자 하는 강한 욕구가 있습니다.",
            "sdt": "유능감 (Competence)",
            "career_hint": "지속적 학습과 역량 개발이 가능한 분야, 또는 타인의 성장을 돕는 역할에 잘 맞습니다.",
        },
    },
    "tech_knowledge": {
        "label": "기술/지식",
        "emoji": "💡",
        "seeds": [
            "ai", "인공지능", "프로그래밍", "코딩", "강의", "알고리즘", "데이터", "기술",
            "과학", "수학", "물리", "화학", "생물", "논문", "개발", "python", "java",
            "머신러닝", "딥러닝", "클라우드", "api", "서버", "it", "컴퓨터",
        ],
        "motivation": {
            "core": "지적 호기심·유능감 욕구",
            "detail": "복잡한 개념을 이해하고 문제를 해결하는 과정에서 만족을 얻습니다.",
            "sdt": "유능감 (Competence) + 자율성 (Autonomy)",
            "career_hint": "기술적 전문성이 요구되는 분야, 또는 복잡한 시스템을 설계·분석하는 역할에 강점이 있습니다.",
        },
    },
    "economy_startup": {
        "label": "경제/창업",
        "emoji": "💰",
        "seeds": [
            "투자", "사업", "재테크", "마케팅", "창업", "부동산", "주식", "돈", "경제", "금융",
            "스타트업", "벤처", "비즈니스", "etf", "코인", "펀드", "배당", "수익", "부자",
            "직장", "연봉", "이직", "취업", "파이프라인",
        ],
        "motivation": {
            "core": "자율성·통제감 욕구",
            "detail": "스스로의 결정으로 미래를 만들고 싶어합니다.",
            "sdt": "자율성 (Autonomy)",
            "career_hint": "의사결정 권한이 있는 직무, 창업, 또는 전략적 기획 역할에 잘 맞습니다.",
        },
    },
    "emotional_healing": {
        "label": "감정/힐링",
        "emoji": "🌱",
        "seeds": [
            "심리", "상담", "힐링", "명상", "감정", "치유", "마음", "위로", "자존감", "관계",
            "mbti", "인간관계", "외로움", "우울", "불안", "치료", "공감",
            "마음챙김", "mindfulness", "요가", "asmr", "잔잔", "편안",
        ],
        "motivation": {
            "core": "정서적 안정·자기 이해 욕구",
            "detail": "자신과 타인의 감정을 깊이 이해하려는 욕구가 강합니다.",
            "sdt": "관계성 (Relatedness)",
            "career_hint": "공감과 정서적 지지가 핵심인 직무 — 상담, 코칭, 교육, 사회복지.",
        },
    },
    "health_fitness": {
        "label": "건강/피트니스",
        "emoji": "💪",
        "seeds": [
            "운동", "헬스", "다이어트", "근육", "식단", "러닝", "요가", "홈트", "건강",
            "pt", "트레이닝", "체력", "단백질", "칼로리", "체중", "필라테스",
            "수면", "회복", "재활", "스포츠", "마라톤",
        ],
        "motivation": {
            "core": "신체 통제감·자기관리 욕구",
            "detail": "자신의 몸과 건강을 스스로 관리하고 발전시키려는 욕구가 강합니다.",
            "sdt": "유능감 (Competence) + 자율성 (Autonomy)",
            "career_hint": "신체와 건강을 다루는 직무, 또는 규율과 루틴이 있는 환경에서 강점을 발휘합니다.",
        },
    },
    "creative_arts": {
        "label": "창작/예술",
        "emoji": "🎨",
        "seeds": [
            "음악", "그림", "영상", "디자인", "작곡", "편집", "창작", "예술", "미술", "사진",
            "일러스트", "웹툰", "만화", "피그마", "포토샵", "premiere", "adobe",
            "촬영", "카메라", "렌즈", "스케치",
        ],
        "motivation": {
            "core": "자기표현·심미적 만족 욕구",
            "detail": "자신만의 방식으로 세상을 표현하고 싶어합니다.",
            "sdt": "자율성 (Autonomy) + 유능감 (Competence)",
            "career_hint": "창의적 표현과 자율성이 보장되는 직무에서 최고의 성과를 냅니다.",
        },
    },
    "culture_society": {
        "label": "문화/사회",
        "emoji": "🌍",
        "seeds": [
            "뉴스", "시사", "역사", "다큐", "정치", "사회", "문화", "철학", "인문",
            "국제", "외교", "세계", "전쟁", "환경", "기후", "인권",
            "다큐멘터리", "documentary", "팩트", "검증",
        ],
        "motivation": {
            "core": "세계관 형성·의미 추구 욕구",
            "detail": "세상이 어떻게 돌아가는지 이해하고 자신만의 관점을 형성하려는 욕구가 있습니다.",
            "sdt": "자율성 (Autonomy)",
            "career_hint": "비판적 사고와 분석이 필요한 직무, 또는 사회적 영향력이 있는 분야.",
        },
    },
}


def classify_titles_to_clusters(titles: list) -> dict:
    cluster_counts = defaultdict(int)
    total = len(titles) or 1

    for title in titles:
        t = title.lower()
        best_cluster = None
        best_score = 0
        for cluster_key, cluster_info in CONTENT_CLUSTERS.items():
            score = sum(1 for seed in cluster_info["seeds"] if seed in t)
            if score > best_score:
                best_score = score
                best_cluster = cluster_key
        if best_cluster:
            cluster_counts[best_cluster] += 1

    result = {}
    for k, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
        if count == 0:
            continue
        cluster = CONTENT_CLUSTERS[k]
        result[k] = {
            "label": cluster["label"],
            "emoji": cluster["emoji"],
            "count": count,
            "ratio": round(count / total, 4),
            "pct": round(count / total * 100, 1),
            "motivation": cluster["motivation"],
        }
    return result


# ============================================================
# 직업 DB 로드
# ============================================================
def load_jobs() -> list:
    with open(JOB_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Profile 메타 정보
# ============================================================
PROFILE_META = {
    "creator": {
        "title": "창작형 성향",
        "riasec": "Artistic(A) — 새로운 콘텐츠를 만들고 자신의 생각을 표현하는 과정에서 동기를 얻는 유형입니다.",
        "scct": "콘텐츠 제작 분야에 대한 자기효능감이 높게 나타납니다.",
        "sdt": "창의적 표현 욕구와 자율성이 높게 나타납니다.",
    },
    "planner": {
        "title": "기획형 성향",
        "riasec": "Investigative(I) + Enterprising(E) — 정보를 분석하고 더 나은 해결책을 설계하는 유형입니다.",
        "scct": "서비스와 사용자 경험에 대한 관심이 높게 나타납니다.",
        "sdt": "문제를 발견하고 개선하는 과정에서 동기를 얻습니다.",
    },
    "marketer": {
        "title": "소통형 성향",
        "riasec": "Enterprising(E) + Social(S) — 사람의 관심과 반응을 이해하는 데 강점을 보입니다.",
        "scct": "트렌드와 대중 반응에 대한 관심이 높게 나타납니다.",
        "sdt": "영향력을 발휘하는 과정에서 동기를 얻는 유형입니다.",
    },
    "startup": {
        "title": "창업형 성향",
        "riasec": "Enterprising(E) — 새로운 아이디어를 실행하고 사람들을 이끄는 역할에 강점이 있습니다.",
        "scct": "비즈니스 모델 설계와 실행에 대한 자기효능감이 높습니다.",
        "sdt": "도전과 자율성을 바탕으로 새로운 가치를 창출할 때 강한 동기를 느낍니다.",
    },
    "education": {
        "title": "성장지원형 성향",
        "riasec": "Social(S) — 타인의 성장과 발전을 돕는 과정에서 만족을 느낍니다.",
        "scct": "지식 전달과 상담 분야에 대한 관심이 높게 나타납니다.",
        "sdt": "사람의 성장을 돕는 과정에서 높은 동기를 얻습니다.",
    },
    "design": {
        "title": "시각창작형 성향",
        "riasec": "Artistic(A) + Investigative(I) — 시각적 문제를 해결하고 아름다움을 설계하는 유형입니다.",
        "scct": "디자인 툴 활용과 창작 활동에 대한 자기효능감이 높게 나타납니다.",
        "sdt": "심미적 표현과 사용자 공감에서 동기를 얻습니다.",
    },
    "lifestyle": {
        "title": "탐험형 성향",
        "riasec": "Enterprising(E) + Artistic(A) — 새로운 경험과 활동을 탐색하는 것을 즐깁니다.",
        "scct": "새로운 경험을 추구하는 성향이 나타납니다.",
        "sdt": "도전과 탐험에서 동기를 얻는 유형입니다.",
    },
    "developer": {
        "title": "문제해결형 성향",
        "riasec": "Investigative(I) — 논리적 사고와 문제 해결을 선호하는 유형입니다.",
        "scct": "기술적 문제 해결에 대한 관심이 높게 나타납니다.",
        "sdt": "복잡한 문제를 해결하는 과정에서 동기를 얻습니다.",
    },
    "general": {
        "title": "탐색형 성향",
        "riasec": "Investigative(I) — 새로운 지식을 탐구하는 유형입니다.",
        "scct": "다양한 분야를 탐색하는 성향이 나타납니다.",
        "sdt": "새로운 배움에서 동기를 얻습니다.",
    },
}


# ============================================================
# 엔드포인트
# ============================================================
@app.route("/")
def home():
    return jsonify({
        "service": "CareerTube Backend",
        "status": "running",
        "version": "3.3.0",
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON 형식의 요청 데이터가 필요합니다."}), 400

    titles = data.get("titles", [])
    channels = data.get("channels", [])

    if not titles:
        return jsonify({"error": "titles 필드가 비어 있습니다."}), 400

    try:
        jobs = load_jobs()
    except FileNotFoundError:
        return jsonify({"error": "job_db.json 파일을 찾을 수 없습니다."}), 500

    # ── 키워드 추출 ───────────────────────────────────────────
    watch_keywords = extract_keywords_smart(titles, top_n=50)

    # 구독 채널명에서 진로 키워드 주입 (수학, 탐구, 사고실험 등)
    channel_injected = inject_channel_keywords(channels)

    # 최종 키워드: 채널 주입 키워드 우선 배치 + 시청 기록 키워드 합산 (중복 제거)
    all_kw_set = set(channel_injected)
    merged_keywords = list(channel_injected)
    for kw in watch_keywords:
        if kw not in all_kw_set:
            merged_keywords.append(kw)
            all_kw_set.add(kw)
    keywords = merged_keywords[:60]

    channel_keywords = []
    for ch in channels:
        # 구독 채널명은 진로 신호 강도가 높으므로 6배 가중치
        channel_keywords += _tokenize_title(ch) * 6

    # ── TF-IDF 직업 매칭 ─────────────────────────────────────
    # 추출된 핵심 키워드를 5배 반복해 직업 매칭 가중치 강화
    keyword_text = " ".join(keywords * 5)
    channel_text = " ".join(channel_keywords)
    # 원제목은 너무 많아서 노이즈가 됨 → 2000개만 샘플링
    title_sample = titles[:2000]
    title_text = " ".join(str(t) for t in title_sample)
    user_text = keyword_text + " " + channel_text + " " + title_text

    job_texts = [" ".join(job.get("keywords", [])) for job in jobs]
    corpus = [user_text] + job_texts

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

    # ── 추천 목록 구성 ────────────────────────────────────────
    recommendations = []
    for idx, score in enumerate(similarities):
        job = jobs[idx]
        recommendations.append({
            "job":               job["job"],
            "description":       job.get("description", ""),
            "keywords":          job.get("keywords", []),
            "profile_type":      job.get("profile_type", "general"),
            "behavior_patterns": job.get("behavior_patterns", []),
            "strengths":         job.get("strengths", []),
            "reason":            job.get("reason", ""),
            "_raw_sim":          float(score),
            "score":             round(min(float(score) * 100, 100.0), 1),
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)

    # ── 점수 정규화: top job → 82%, 나머지 비례 ──────────────
    # cosine similarity는 데이터 특성상 절대값이 낮게 나오므로
    # 상대 매칭 강도를 82% 기준으로 표현
    raw_max = recommendations[0]["_raw_sim"] if recommendations else 0
    if raw_max > 0:
        for r in recommendations:
            r["score"] = round((r["_raw_sim"] / raw_max) * 82.0, 1)
    for r in recommendations:
        del r["_raw_sim"]

    top3 = recommendations[:3]

    # ── Top 1 직업 상세 정보 ──────────────────────────────────
    top_job_name = top3[0]["job"] if top3 else ""
    top_job_data = next((j for j in jobs if j["job"] == top_job_name), None)

    if top_job_data:
        majors            = top_job_data.get("majors", [])
        skills            = top_job_data.get("skills", [])
        roadmap           = top_job_data.get("roadmap", [])
        profile_type      = top_job_data.get("profile_type", "general")
        behavior_patterns = top_job_data.get("behavior_patterns", [])
        strengths         = top_job_data.get("strengths", [])
        reason            = top_job_data.get("reason", "")
    else:
        majors = skills = roadmap = []
        profile_type = "general"
        behavior_patterns = strengths = []
        reason = ""

    meta = PROFILE_META.get(profile_type, PROFILE_META["general"])
    keyword_str = ", ".join(keywords[:10])

    career_report = {
        "top3_detail": (
            f"당신은 {meta['title']}입니다.\n\n"
            f"주요 관심 키워드: {keyword_str}\n\n"
            f"주요 행동 패턴: {', '.join(behavior_patterns)}\n\n"
            f"추천 이유: {reason}\n\n"
            f"예상 강점: {', '.join(strengths)}"
        ),
        "major_detail": (
            f"추천 직업인 '{top_job_name}' 분야로 진출하기 위해서는 "
            f"{', '.join(majors) if majors else '관련 전공'} 관련 학습이 도움이 됩니다."
        ),
        "skill_detail": (
            "해당 분야에서 중요하게 요구되는 역량:\n"
            + "\n".join(f"- {s}" for s in skills)
        ) if skills else "역량 정보를 불러오는 중입니다.",
        "roadmap_detail": (
            "추천 진로 준비 과정:\n"
            + "\n".join(f"  {i+1}단계. {step}" for i, step in enumerate(roadmap))
        ) if roadmap else "로드맵 정보를 불러오는 중입니다.",
        "riasec":   meta["riasec"],
        "scct":     meta["scct"],
        "sdt":      meta["sdt"],
        "aptitude": f"강점 역량: {', '.join(strengths)}\n성향 유형: {meta['title']}",
    }

    analysis = (
        f"[CareerTube AI 진로 분석]\n\n"
        f"추천 직업 1순위: {top_job_name}\n"
        f"주요 관심 키워드: {keyword_str}\n\n"
        f"추천 학과: {', '.join(majors[:3]) if majors else '관련 전공'}\n\n"
        f"우선 개발 역량: {', '.join(skills[:3]) if skills else '핵심 역량'}"
    )

    content_clusters = classify_titles_to_clusters(titles)

    # ── Google Sheets 저장 ────────────────────────────────────
    log_payload = {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "keywords":      ",".join(keywords[:20]),
        "job1":          top3[0]["job"] if len(top3) > 0 else "",
        "job2":          top3[1]["job"] if len(top3) > 1 else "",
        "job3":          top3[2]["job"] if len(top3) > 2 else "",
        "title_count":   len(titles),
        "channel_count": len(channels),
        "dream_job": "",
        "match": "",
    }
    try:
        requests.get(
            GOOGLE_SHEET_URL,
            params={"action": "insert", "table": "logs", "data": json.dumps(log_payload)},
            timeout=5,
        )
    except Exception as e:
        print(f"[Google Sheets] 저장 실패: {e}")

    return jsonify({
        "keywords":         keywords[:60],
        "keyword_method":   "Interest Intensity Score + Bigram + 해시태그 고유명사 필터",
        "recommendations":  top3,
        "majors":           majors,
        "skills":           skills,
        "roadmap":          roadmap,
        "analysis":         analysis,
        "career_report":    career_report,
        "content_clusters": content_clusters,
        "raw_titles":       titles[:100],
    })


# ============================================================
# 서버 실행
# ============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))

    debug = False  # 🚨 Render에서는 무조건 False

    print(f"CareerTube Backend running on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug
    )
