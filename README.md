# careertube
# CareerTube 🎯

**YouTube 시청 기록 기반 진로 추천 서비스**

유튜브 시청 기록과 구독 채널을 분석하여 사용자의 관심사를 추출하고, TF-IDF + 코사인 유사도를 통해 적합한 진로를 추천합니다.

---

## 🚀 바로 사용하기 (배포 버전)

> **별도 설치 없이 바로 실행 가능합니다.**

1. `backend/CareerTube.html` 파일을 브라우저에서 열기
2. YouTube 시청 기록(`시청기록.html`)과 구독 정보(`구독정보.csv`) 파일 업로드
3. **분석하기** 버튼 클릭

백엔드는 Render에 배포되어 있어 별도 서버 실행이 필요 없습니다.  
※ 무료 플랜 특성상 첫 요청 시 30~60초 대기 시간이 있을 수 있습니다.

---

## 📁 프로젝트 구조

```
backend/
├── CareerTube.html      # 프론트엔드 (단일 HTML 파일)
├── app.py               # Flask 백엔드 API
├── job_db.json          # 직업 데이터베이스 (39개 직업)
├── requirements.txt     # Python 의존성
└── render.yaml          # Render 배포 설정
```

---

## 🛠️ 로컬 실행 방법

### 1. 환경 설정

```bash
cd backend
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
python app.py
```

서버가 `http://127.0.0.1:5000` 에서 실행됩니다.

### 3. 프론트엔드 연결

`CareerTube.html` 파일의 `BACKEND_URL`을 로컬로 변경:

```html
<!-- CareerTube.html 상단 스크립트에서 -->
const BACKEND_URL = "http://127.0.0.1:5000";  // 로컬 실행 시
// const BACKEND_URL = "https://careertube-api.onrender.com";  // 배포 버전
```

이후 `CareerTube.html`을 브라우저에서 열어서 사용합니다.

---

## 📊 분석 파일 준비 방법 (Google 테이크아웃)

두 파일 모두 **[Google 테이크아웃](https://takeout.google.com/)** 에서 한 번에 받을 수 있습니다.

### 1단계 — 테이크아웃 설정

1. [https://takeout.google.com](https://takeout.google.com) 접속
2. **"모두 선택 해제"** 클릭 → 목록 맨 아래로 스크롤
3. **"YouTube 및 YouTube Music"** 항목만 체크

### 2단계 — YouTube 데이터 세부 선택

4. "YouTube 및 YouTube Music" 옆 **"여러 형식"** 클릭
   - **기록** → 형식을 **HTML**로 변경 (기본값은 JSON)
5. **"포함된 모든 YouTube 데이터"** 클릭
   - **"시청 기록"** 과 **"구독"** 만 체크, 나머지 해제

### 3단계 — 내보내기 실행

6. 하단 **"다음 단계"** 클릭
7. 내보내기 방법: 이메일로 받기 / .zip / 1회 내보내기 선택 후 **"내보내기 만들기"**
8. 이메일로 다운로드 링크 수신 (보통 수 분 ~ 수십 분 소요)

### 4단계 — 파일 추출

9. 다운로드한 `.zip` 압축 해제
10. 아래 경로에서 파일 찾기:
    - 시청 기록: `Takeout/YouTube 및 YouTube Music/기록/시청 기록.html`
    - 구독 정보: `Takeout/YouTube 및 YouTube Music/구독/구독정보.csv`
11. 두 파일을 CareerTube에 업로드

---

## ⚙️ 기술 스택

| 구분 | 기술 |
|------|------|
| Frontend | HTML / CSS / JavaScript (단일 파일) |
| Backend | Python, Flask |
| 분석 알고리즘 | TF-IDF, 코사인 유사도 (scikit-learn) |
| 배포 | Render (백엔드), 브라우저 직접 실행 (프론트엔드) |

---

## 🔍 추천 알고리즘 개요

1. **키워드 추출**: 시청 기록 제목에서 TF × log(1+DF) × (1+Coverage) 점수로 의미 있는 키워드 추출
2. **채널 분석**: 구독 채널명으로부터 진로 관련 키워드 주입 (6배 가중치)
3. **직업 매칭**: 추출된 키워드와 39개 직업 프로필 간 코사인 유사도 계산
4. **점수 정규화**: 1위 직업을 82%로 정규화하여 신뢰도 있는 수치 표시

---

## 👤 개발자

**KanghanLee**

