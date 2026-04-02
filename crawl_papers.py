"""
건강식품·다이어트 관련 PMC 오픈액세스 논문 PDF 크롤러
- PMC 아티클 페이지에서 PDF 경로를 파싱
- 쿠키·세션을 유지하며 실제 PDF를 다운로드
"""
import requests
import re
import os
import json
import time
from pathlib import Path

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://pmc.ncbi.nlm.nih.gov/",
})

BASE_DIR = Path("C:/dev/healthfood_chatbot/data/raw/papers")
META_FILE = Path("C:/dev/healthfood_chatbot/data/raw/papers/download_metadata.json")

# ── 논문 목록 ────────────────────────────────────────────────────────
PAPERS = {
    "health_food": [
        {
            "pmc_id": "PMC12011469",
            "filename": "probiotics_functional_food_beverages_2025",
            "title": "Progress in Probiotic Science: Prospects of Functional Probiotic-Based Foods and Beverages",
            "year": 2025,
            "topic": "프로바이오틱스 기반 기능성 식품·음료 전망",
        },
        {
            "pmc_id": "PMC12073198",
            "filename": "probiotics_prebiotics_human_health_2025",
            "title": "The Impact of Probiotics, Prebiotics, and Functional Foods on Human Health",
            "year": 2025,
            "topic": "프로바이오틱스·프리바이오틱스·기능성식품 건강 영향",
        },
        {
            "pmc_id": "PMC11315846",
            "filename": "probiotic_strains_functional_foods_2024",
            "title": "Harnessing the Power of Probiotic Strains in Functional Foods",
            "year": 2024,
            "topic": "기능성식품 내 프로바이오틱스 균주 활용",
        },
        {
            "pmc_id": "PMC12628397",
            "filename": "omega3_EPA_DHA_cardiovascular_2025",
            "title": "N-3 Fatty Acids (EPA and DHA) and Cardiovascular Health",
            "year": 2025,
            "topic": "오메가-3 EPA·DHA와 심혈관 건강",
        },
        {
            "pmc_id": "PMC11090157",
            "filename": "omega3_dementia_prevention_2024",
            "title": "An Analysis of Omega-3 Clinical Trials and Personalized Supplementation for Dementia Prevention",
            "year": 2024,
            "topic": "오메가-3 임상시험 분석 및 치매 예방",
        },
        {
            "pmc_id": "PMC12365556",
            "filename": "red_ginseng_mitochondrial_immunosenescence_2025",
            "title": "Red Ginseng Extract Enhances Mitochondrial Function and Alleviates Immunosenescence in T Cells",
            "year": 2025,
            "topic": "홍삼 추출물의 미토콘드리아 기능 강화 및 면역노화 완화",
        },
        {
            "pmc_id": "PMC12125682",
            "filename": "red_ginseng_anti_fatigue_RCT_2025",
            "title": "Anti-Fatigue Effects of Korean Red Ginseng Extract in Healthy Japanese Adults: A Randomized, Double-Blind, Placebo-Controlled Study",
            "year": 2025,
            "topic": "홍삼 추출물 항피로 효과 이중맹검 RCT",
        },
        {
            "pmc_id": "PMC12365502",
            "filename": "red_ginseng_chronic_pancreatitis_2025",
            "title": "Korean Red Ginseng Supplements Improve Quality of Life in Patients with Mild Chronic Pancreatitis",
            "year": 2025,
            "topic": "홍삼 보충제의 만성 췌장염 삶의 질 개선",
        },
    ],
    "diet": [
        {
            "pmc_id": "PMC12583794",
            "filename": "korean_obesity_guidelines_2024",
            "title": "2024 Clinical Practice Guidelines for Diagnosis and Treatment of Overweight and Obesity (Korean Society for the Study of Obesity)",
            "year": 2024,
            "topic": "2024 한국 비만 진단·치료 임상 가이드라인 (대한비만학회)",
        },
        {
            "pmc_id": "PMC11890254",
            "filename": "ketogenic_diet_obesity_review_2025",
            "title": "Ketogenic Diet Intervention for Obesity Weight-Loss: A Narrative Review",
            "year": 2025,
            "topic": "케토제닉 다이어트 비만 체중 감량 효과 리뷰",
        },
        {
            "pmc_id": "PMC11884964",
            "filename": "behavioral_weight_management_meta_analysis_2025",
            "title": "The Impact of Behavioral Weight Management Interventions on Eating Behavior: A Systematic Review and Meta-Analysis",
            "year": 2025,
            "topic": "행동 기반 체중 관리 중재의 식이 행동 영향 메타분석",
        },
        {
            "pmc_id": "PMC11844017",
            "filename": "medical_management_obesity_2025",
            "title": "Medical Management of Obesity: Current Trends and Future Perspectives",
            "year": 2025,
            "topic": "비만의 약물·의학적 관리 현황 및 전망",
        },
        {
            "pmc_id": "PMC11996779",
            "filename": "public_health_obesity_strategies_2025",
            "title": "Tackling the Complexity of Obesity Through Public Health Strategies",
            "year": 2025,
            "topic": "비만 해결을 위한 공중 보건 전략",
        },
        {
            "pmc_id": "PMC11623039",
            "filename": "guideline_directed_obesity_treatment_2024",
            "title": "A Guideline-Directed Approach to Obesity Treatment",
            "year": 2024,
            "topic": "가이드라인 기반 비만 치료 접근법",
        },
    ],
}


def get_pdf_path(pmc_id: str) -> str | None:
    """PMC 아티클 페이지에서 PDF 상대경로 파싱"""
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
    try:
        r = SESSION.get(url, timeout=20)
        match = re.search(r'href="(pdf/[^"]+\.pdf)"', r.text)
        return match.group(1) if match else None
    except Exception as e:
        print(f"  [!] 페이지 접근 실패: {e}")
        return None


def download_pdf(pmc_id: str, pdf_rel: str, out_path: Path) -> bool:
    """실제 PDF 다운로드 (쿠키 유지 세션)"""
    # 먼저 아티클 페이지를 방문해 쿠키를 세팅
    article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
    SESSION.get(article_url, timeout=20)
    time.sleep(1)

    pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/{pdf_rel}"
    SESSION.headers.update({
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": article_url,
    })
    try:
        r = SESSION.get(pdf_url, timeout=60, stream=True)
        # 첫 바이트가 %PDF 이면 진짜 PDF
        content = b""
        for chunk in r.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 4:
                break
        if not content.startswith(b"%PDF"):
            return False
        # 나머지 내용도 수신
        for chunk in r.iter_content(chunk_size=65536):
            content += chunk
        out_path.write_bytes(content)
        return True
    except Exception as e:
        print(f"  [!] 다운로드 오류: {e}")
        return False


def main():
    metadata = {"papers": []}
    total, success = 0, 0

    for category, papers in PAPERS.items():
        save_dir = BASE_DIR / category
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"카테고리: {category.upper()} ({len(papers)}편)")
        print("=" * 60)

        for paper in papers:
            pmc_id = paper["pmc_id"]
            filename = paper["filename"]
            out_path = save_dir / f"{filename}.pdf"
            total += 1

            print(f"\n[{total}] {pmc_id} - {filename}")
            print(f"  제목: {paper['title'][:70]}...")

            # PDF 경로 탐색
            pdf_rel = get_pdf_path(pmc_id)
            if not pdf_rel:
                print("  [SKIP] PDF 경로를 찾을 수 없음")
                status = "no_pdf_path"
            else:
                print(f"  PDF 경로: {pdf_rel}")
                ok = download_pdf(pmc_id, pdf_rel, out_path)
                if ok:
                    size_kb = out_path.stat().st_size // 1024
                    print(f"  [OK] 저장 완료 → {out_path.name} ({size_kb} KB)")
                    status = "success"
                    success += 1
                else:
                    print("  [FAIL] PDF 내용 수신 실패 (HTML 반환)")
                    status = "html_redirect"
                    if out_path.exists():
                        out_path.unlink()

            metadata["papers"].append({
                **paper,
                "category": category,
                "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/",
                "pdf_rel_path": pdf_rel,
                "local_path": str(out_path) if status == "success" else None,
                "status": status,
            })
            time.sleep(1.5)

    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"완료: {success}/{total}편 다운로드 성공")
    print(f"메타데이터: {META_FILE}")


if __name__ == "__main__":
    main()
