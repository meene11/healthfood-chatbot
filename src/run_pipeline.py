"""
전체 파이프라인 실행 스크립트
1. 전체 데이터(PDF+TXT+JSON) 파싱 → 청크 JSON
2. 임베딩 생성 → Supabase 업로드
"""

# 1단계: 전체 데이터 파싱
print("\n[1/2] 전체 데이터 파싱 시작...")
import parse_all_data
parse_all_data.main()

# 2단계: Supabase 업로드
print("\n[2/2] Supabase 업로드 시작...")
import upload_to_supabase
upload_to_supabase.main()

print("\n" + "="*60)
print("파이프라인 완료! 챗봇 실행: python src/chatbot.py")
print("="*60)
