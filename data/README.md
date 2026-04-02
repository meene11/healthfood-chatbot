# 건강식품 챗봇 데이터 구조

## 디렉토리 구조

```
data/
├── raw/                                        # 원천(raw) 데이터 — 직접 수집·크롤링한 데이터
│   ├── health_foods.json                       # 건강식품 10종 상세 정보 (수동 작성)
│   ├── categories.json                         # 카테고리 및 검색 키워드
│   ├── faq.json                                # 자주 묻는 질문 5개
│   ├── crawled_kdca_health_functional_food.json  # 질병관리청 건강기능식품 개요
│   ├── crawled_nih_omega3.json                 # NIH 오메가-3 팩트시트 (영문)
│   ├── crawled_nih_vitamin_d.json              # NIH 비타민D 팩트시트 (영문)
│   ├── crawled_pmc_probiotics_research.json    # PMC 프로바이오틱스 리뷰 논문
│   └── crawled_search_summary.json             # 크롤링 소스 인덱스 (출처 목록)
│
└── generated/                                  # 생성(generated) 데이터 — raw를 정제·가공한 데이터
    ├── unified_knowledge_base.json             # 통합 지식베이스 (챗봇 응답용)
    ├── chatbot_qa_pairs.json                   # Q&A 페어 10개 (RAG 학습/검색용)
    └── ingredient_index.json                   # 키워드→kb_id 검색 인덱스
```

## Raw 데이터 설명

| 파일 | 출처 | 설명 |
|------|------|------|
| `health_foods.json` | 수동 작성 | 건강식품 10종 성분·효능·복용법·주의사항 |
| `categories.json` | 수동 작성 | 카테고리 10개 + 검색 키워드 |
| `faq.json` | 수동 작성 | 자주 묻는 질문 5개 |
| `crawled_kdca_health_functional_food.json` | 질병관리청 / 식약처 | 건강기능식품 정의·기능성 분류·주의사항 |
| `crawled_nih_omega3.json` | NIH ODS (미국 국립보건원) | 오메가-3 성분·임상근거·섭취량·약물상호작용 |
| `crawled_nih_vitamin_d.json` | NIH ODS (미국 국립보건원) | 비타민D 유형·결핍증·효능·독성 기준치 |
| `crawled_pmc_probiotics_research.json` | PMC / Frontiers in Nutrition | 프로바이오틱스 차세대 균주·임상근거·작용기전 |
| `crawled_search_summary.json` | 크롤링 메타 | 모든 수집 소스의 출처 URL·파일 매핑 |

## Generated 데이터 설명

raw 데이터를 정제·통합하여 챗봇이 바로 활용할 수 있도록 가공한 데이터입니다.

| 파일 | 설명 |
|------|------|
| `unified_knowledge_base.json` | 5개 주제별 통합 지식베이스 (효능·근거수준·주의사항·출처) |
| `chatbot_qa_pairs.json` | 예상 Q&A 10쌍 (RAG 검색 또는 파인튜닝 학습 데이터) |
| `ingredient_index.json` | 키워드→지식베이스 ID 매핑 (빠른 검색용) |

## 데이터 흐름

```
웹 크롤링 / 수동 작성
        ↓
   raw/ 저장 (원본 보존)
        ↓
   정제·통합·Q&A 생성
        ↓
   generated/ 저장 (챗봇 사용)
```
