"""
건강식품 챗봇 논문 전체 크롤러
- raw/papers/diet/           : 다이어트 논문 (기존 + 신규)
- raw/papers/health_food/    : 건강식품 효능·부작용 논문
- raw/papers/diet_nutrition/ : 다이어트 영양소 논문 (신규)
"""
import requests, time, json
from pathlib import Path

S = requests.Session()
S.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Referer": "https://www.frontiersin.org/",
})

BASE = Path("C:/dev/healthfood_chatbot/data/raw/papers")
BASE.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 카테고리별 논문 목록
# ─────────────────────────────────────────────────────────────
PAPERS = {

  # ══════════════════════════════════════════════════════════
  # A. 다이어트 (기존 1차 복구)
  # ══════════════════════════════════════════════════════════
  "diet": [
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1677496/pdf",
     "plant_based_diet_weight_loss_RCT_2025",
     "VEGPREV: 4가지 식물성 식단의 체중 감량 효과 비교 RCT"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1548609/pdf",
     "metabolic_health_dietary_intervention_2025",
     "TOWARD 대사 건강 식이 중재 프로그램"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1458353/pdf",
     "glycemic_index_diet_insulin_resistance_2025",
     "혈당지수 식단과 인슐린 저항성"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1482854/pdf",
     "health_at_every_size_diet_review_2024",
     "HAES 체중 중립적 식이 접근법"),
    ("https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1671649/pdf",
     "digital_therapeutic_weight_management_2025",
     "디지털 치료 기기 기반 체중 관리"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1695412/pdf",
     "diet_obesity_nutrition_intervention_2025",
     "비만 영양 중재 효과"),
    # 기존 2차 복구
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1524125/pdf",
     "intermittent_fasting_cardiovascular_risk_2025",
     "간헐적 단식과 심혈관 위험"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1664412/pdf",
     "time_restricted_eating_body_composition_2025",
     "TRE가 체성분·대사에 미치는 영향 메타분석"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1439473/pdf",
     "5_2_fasting_vs_calorie_restriction_fatty_liver_2024",
     "5:2 단식 vs 매일 칼로리 제한 (지방간 RCT)"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1658691/pdf",
     "ketogenic_diet_metabolic_syndrome_2025",
     "케토제닉 식단과 대사증후군"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1516086/pdf",
     "low_carb_diet_T2DM_meta_analysis_2024",
     "저탄수화물 식단과 제2형 당뇨 메타분석"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1342787/pdf",
     "intermittent_fasting_gut_microbiota_2024",
     "간헐적 단식과 장내 미생물 체계적 리뷰"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1393292/pdf",
     "intermittent_fasting_obesity_microbiome_2024",
     "단식·비만·장내 미생물 상관관계"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1647740/pdf",
     "fiber_polyphenol_diet_gut_microbiome_2025",
     "식이섬유·폴리페놀 식단과 장내 미생물 건강 효과"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1634545/pdf",
     "dietary_patterns_metabolic_syndrome_network_meta_2025",
     "6가지 식이 패턴 대사증후군 네트워크 메타분석"),
    ("https://public-pages-files-2025.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1325894/pdf",
     "intermittent_fasting_regimens_comparison_2024",
     "간헐적 단식 방식별 비교 리뷰"),
    ("https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2025.1675155/pdf",
     "gut_microbiota_fat_deposition_mechanisms_2025",
     "장내 미생물이 체지방 축적에 미치는 기전"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1376098/pdf",
     "low_carbohydrate_diet_expert_consensus_2024",
     "저탄수화물 식단 전문가 합의문"),
  ],

  # ══════════════════════════════════════════════════════════
  # B. 건강식품 효능·부작용 (기존 복구)
  # ══════════════════════════════════════════════════════════
  "health_food": [
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1650883/pdf",
     "probiotics_clinical_applications_trends_2025",
     "프로바이오틱스 임상 적용 현황 및 동향"),
    ("https://www.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2025.1561047/pdf",
     "probiotics_synbiotics_IBD_treatment_2025",
     "프로바이오틱스·신바이오틱스의 IBD 치료"),
    ("https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2024.1487641/pdf",
     "probiotics_human_health_comprehensive_review_2024",
     "프로바이오틱스와 인체 건강 종합 리뷰"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1588421/pdf",
     "omega3_EPA_DHA_exercise_physiology_2025",
     "오메가-3 EPA·DHA와 운동 생리 (6주 RCT)"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1575323/pdf",
     "omega3_gut_microbiota_ratio_2025",
     "오메가-3와 장내 미생물 및 오메가-6/3 비율"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1497207/pdf",
     "nutritional_supplements_global_trends_2025",
     "2000-2024 글로벌 건강보조식품 연구 동향"),
    ("https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2025.1601204/pdf",
     "curcumin_health_outcomes_umbrella_review_2025",
     "커큐민 건강 효과 우산 리뷰"),
    ("https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2025.1509045/pdf",
     "curcumin_pharmacological_effects_clinical_2025",
     "커큐민 약리 효과·제형·임상 현황"),
    ("https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1603018/pdf",
     "curcumin_bioavailability_immune_health_2025",
     "커큐민 생체이용률·면역 건강 종합 리뷰"),
    ("https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2025.1540255/pdf",
     "ginseng_fatty_liver_pharmacology_2025",
     "인삼·진세노사이드의 비알코올성 지방간 기전"),
    ("https://www.frontiersin.org/journals/cellular-and-infection-microbiology/articles/10.3389/fcimb.2025.1455735/pdf",
     "probiotics_adverse_events_pharmacovigilance_2025",
     "FDA FAERS 기반 프로바이오틱스 부작용 약물감시"),
    ("https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2025.1594975/pdf",
     "weight_loss_supplement_adulteration_safety_2025",
     "체중 감량 천연 보충제 불법 첨가물 실태"),
    ("https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2025.1654337/pdf",
     "supplement_drug_interaction_older_adults_2025",
     "노인 건강보조식품·약물 상호작용 실태"),
    ("https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1657005/pdf",
     "drug_herb_interaction_review_2025",
     "약물-허브 보충제 상호작용 임상 리뷰"),
    ("https://public-pages-files-2025.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2025.1507812/pdf",
     "supplement_pharmacodynamics_toxicity_review_2025",
     "보충제 약동학·독성·상호작용 리뷰"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1583276/pdf",
     "supplement_efficacy_safety_clinical_2025",
     "임상 집단에서 보충제 효능·안전성"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1380010/pdf",
     "food_processing_medication_interaction_2024",
     "식품 가공과 약물 상호작용"),
  ],

  # ══════════════════════════════════════════════════════════
  # C. 다이어트 영양소 (신규 — 기존과 중복 없음)
  # ══════════════════════════════════════════════════════════
  "diet_nutrition": [
    # 단백질
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1671286/pdf",
     "dietary_protein_amino_acids_obesity_2025",
     "식이 단백질·아미노산이 비만에 미치는 영향"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1547325/pdf",
     "protein_intake_muscle_sarcopenia_elderly_2025",
     "노인 근감소증에서 단백질 섭취의 역할"),
    ("https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2024.1412182/pdf",
     "animal_plant_protein_weight_glycemic_control_2024",
     "동물성·식물성 단백질이 체중·혈당 조절에 미치는 영향"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1370737/pdf",
     "protein_supplemented_VLCD_weight_loss_Korea_2024",
     "단백질 보충 초저열량식이 체중 감량 RCT (한국)"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1619543/pdf",
     "high_protein_diet_body_composition_2025",
     "고단백 식단이 체성분에 미치는 영향"),
    # 미량 영양소
    ("https://public-pages-files-2025.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1363181/pdf",
     "micronutrients_caloric_restriction_fasting_2024",
     "칼로리 제한·단식 중 필수 미량 영양소 요구량"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1646750/pdf",
     "micronutrient_bioavailability_factors_2025",
     "미량 영양소 생체이용률: 개념·영향 인자·개선 전략"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1514681/pdf",
     "magnesium_diet_obesity_metabolism_2025",
     "식이 마그네슘 섭취와 비만·대사"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1563604/pdf",
     "magnesium_vitamin_D_E_coingestion_2025",
     "마그네슘·비타민D·E 복합 섭취 효과"),
    # 식이섬유·저항성 전분
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1369950/pdf",
     "resistant_starch_health_impact_review_2024",
     "저항성 전분의 건강 영향 및 가공 과제 리뷰"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1655664/pdf",
     "resistant_starch_metabolic_syndrome_meta_analysis_2025",
     "저항성 전분과 대사증후군 관련 지표 메타분석"),
    ("https://public-pages-files-2025.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1510564/pdf",
     "dietary_fiber_overall_health_2024",
     "식이섬유가 전반적 건강에 미치는 영향"),
    # 폴리페놀·항산화
    ("https://public-pages-files-2025.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1393575/pdf",
     "polyphenols_anti_obesity_mechanisms_2024",
     "폴리페놀의 항비만 효과: 분자 기전 종합 리뷰"),
    ("https://public-pages-files-2025.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1376508/pdf",
     "polyphenols_body_fat_glucose_metabolism_2024",
     "폴리페놀이 체지방·혈당 대사에 미치는 영향"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1496582/pdf",
     "tea_anti_obesity_bibliometric_2004_2024",
     "차(Tea) 항비만 연구 트렌드 서지계량 분석 (2004-2024)"),
    # BCAA·아미노산
    ("https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1643231/pdf",
     "BCAA_insulin_resistance_type2_diabetes_2025",
     "분지사슬아미노산(BCAA)과 인슐린 저항성·제2형 당뇨"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1709867/pdf",
     "BCAA_muscle_brain_energy_aging_2025",
     "BCAA가 근육-뇌 대사 축·에너지 대사·노화에 미치는 영향"),
    # 크로노뉴트리션·식사 타이밍
    ("https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2024.1359772/pdf",
     "meal_timing_obesity_metabolic_disease_2024",
     "식사 타이밍과 비만·대사 질환"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1663559/pdf",
     "dietary_intake_clock_gene_expression_2025",
     "식이 섭취와 시계 유전자 발현의 연관성"),
    # 칼로리 제한·에너지 균형
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1493954/pdf",
     "caloric_restriction_vs_isocaloric_macronutrient_2024",
     "칼로리 제한 vs 등칼로리 식이: 다량 영양소 조성 비교"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1579024/pdf",
     "exercise_caloric_restriction_body_composition_meta_2025",
     "칼로리 제한 중 운동 방식별 체성분 변화 네트워크 메타분석"),
    ("https://public-pages-files-2025.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2025.1584890/pdf",
     "caloric_restriction_mental_health_2025",
     "칼로리 제한이 정신 건강에 미치는 영향"),
    ("https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1666688/pdf",
     "nutritional_patterns_metabolic_psychological_overweight_2025",
     "과체중·비만 청년의 영양 패턴·대사·심리 상태 네트워크 연구"),
  ],
}


def download(url: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 50_000:
        return True  # 이미 존재하면 스킵
    try:
        r = S.get(url, timeout=60, stream=True)
        data = b""
        for chunk in r.iter_content(65536):
            data += chunk
        if data[:4] == b"%PDF":
            out.write_bytes(data)
            return True
    except Exception as e:
        print(f"    ERROR: {e}")
    return False


def main():
    results = []
    total = success = fail = skip = 0

    for cat, papers in PAPERS.items():
        cat_dir = BASE / cat
        cat_dir.mkdir(exist_ok=True)
        print(f"\n{'='*65}")
        print(f" [{cat.upper()}]  {len(papers)}편")
        print("="*65)

        for url, name, desc in papers:
            out = cat_dir / f"{name}.pdf"
            total += 1

            if out.exists() and out.stat().st_size > 50_000:
                kb = out.stat().st_size // 1024
                print(f"  [SKIP] {name} ({kb}KB)")
                skip += 1
                results.append({"cat": cat, "name": name, "status": "skip", "kb": kb})
                continue

            print(f"  [DL]  {name}")
            ok = download(url, out)
            if ok:
                kb = out.stat().st_size // 1024
                print(f"        → OK ({kb}KB)  {desc}")
                success += 1
                results.append({"cat": cat, "name": name, "status": "ok", "kb": kb,
                                 "url": url, "desc": desc})
            else:
                print(f"        → FAIL")
                fail += 1
                results.append({"cat": cat, "name": name, "status": "fail", "url": url})
            time.sleep(1.2)

    # 결과 저장
    out_json = Path("C:/dev/healthfood_chatbot/data/generated/papers_summary/download_log.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*65}")
    print(f" 완료: 성공 {success} | 스킵 {skip} | 실패 {fail} / 전체 {total}")
    print(f" 저장 위치: {BASE}")
    print("="*65)


if __name__ == "__main__":
    main()
