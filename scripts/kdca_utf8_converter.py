"""
질병관리청 감염병 데이터 UTF-8 인코딩 변환기
실제 CSV 파일들을 한글 헤더와 함께 깔끔한 UTF-8 형태로 변환
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

class KDCAEncodingFixer:
    def __init__(self):
        """초기화 및 한글 매핑 설정"""
        os.makedirs('data/kdca', exist_ok=True)
        os.makedirs('raw_data/kdca', exist_ok=True)
        
        # 지역명 매핑 (캡처 이미지 기준)
        self.regions = [
            '전국', '서울', '부산', '대구', '인천', '광주', '대전', '울산',
            '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '세종'
        ]
        
        # 감염병 컬럼명 (캡처 이미지 기준으로 정확히 매핑)
        self.disease_columns = [
            # 제1급 감염병
            '에볼라바이러스병', '마버그열', '라사열', '크리미안콩고출혈열', '남아메리카출혈열',
            '리프트밸리열', '두창', '페스트', '탄저', '보툴리눔독소증', '야토병',
            '신종호흡기증후군', '중동호흡기증후군', '동물인플루엔자인체감염증', '신종인플루엔자',
            '디프테리아', '수두',
            
            # 제2급 감염병  
            '홍역', '콜레라', '장티푸스', '파라티푸스', '세균성이질', '장출혈성대장균감염증',
            'A형간염', '백일해', '유행성이하선염', '풍진', '폴리오', '수막구균감염증',
            '헤모필루스인플루엔자균감염증', '폐렴구균감염증', '한센병', '성홍열',
            '반코마이신내성황색포도알균(VRSA)감염증', '카바페넴내성장내세균속균종(CRE)감염증',
            
            # 제3급 감염병
            '결핵', '말라리아', '레지오넬라증', '비브리오패혈증', '발진열', '발진티푸스',
            '쯔쯔가무시증', '렙토스피라증', '브루셀라증', '공수병', '신증후군출혈열',
            '후천성면역결핍증(AIDS)', '매독', '크로이츠펠트-야콥병(CJD)', '황열', '뎅기열',
            '웨스트나일열', '라임병', '진드기매개뇌염', '유비저', '치쿤구니야열'
        ]
        
        print("✅ 질병관리청 인코딩 변환기 초기화 완료")
        print(f"📍 지역: {len(self.regions)}개")
        print(f"📊 감염병: {len(self.disease_columns)}개")
    
    def read_broken_csv(self, filepath):
        """인코딩이 깨진 CSV 파일 읽기"""
        encodings = ['euc-kr', 'cp949', 'utf-8', 'latin-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, header=None)
                print(f"✅ {encoding}로 읽기 성공: {filepath}")
                return df
            except Exception as e:
                continue
        
        print(f"❌ 모든 인코딩 시도 실패: {filepath}")
        return None
    
    def convert_to_utf8_with_headers(self, filepath, year):
        """깨진 CSV를 한글 헤더가 있는 UTF-8 CSV로 변환"""
        print(f"\n📋 {year}년 데이터 변환 중...")
        
        # 원본 파일 읽기
        df = self.read_broken_csv(filepath)
        if df is None:
            return None
        
        print(f"원본 크기: {df.shape}")
        
        # 새로운 DataFrame 생성 (한글 헤더 포함)
        # 첫 번째 컬럼은 지역명, 나머지는 감염병별 발생건수
        
        # 지역 개수만큼 행 확보 (전국 포함 18개)
        num_regions = min(len(df), len(self.regions))
        
        # 컬럼 헤더 생성
        headers = ['지역'] + [f'감염병_{i+1:02d}' for i in range(df.shape[1] - 1)]
        
        # 데이터 변환
        new_data = []
        for i in range(num_regions):
            row_data = [self.regions[i]]  # 지역명
            
            # 숫자 데이터 추가 (원본 1번 컬럼부터)
            if i < len(df):
                original_row = df.iloc[i]
                for j in range(1, len(original_row)):
                    try:
                        val = float(original_row.iloc[j]) if pd.notna(original_row.iloc[j]) else 0
                        row_data.append(int(val) if val >= 0 else 0)
                    except:
                        row_data.append(0)
            
            new_data.append(row_data)
        
        # 새로운 DataFrame 생성
        result_df = pd.DataFrame(new_data, columns=headers)
        
        # UTF-8로 저장
        output_path = f'data/kdca/{year}_감염병_UTF8.csv'
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"✅ 변환 완료: {output_path}")
        print(f"새 크기: {result_df.shape}")
        
        # 샘플 데이터 확인
        if len(result_df) > 0:
            print(f"샘플: {result_df.iloc[1]['지역']} = {result_df.iloc[1, 17:22].sum():.0f}건")
        
        return result_df
    
    def convert_all_years(self):
        """전체 연도 데이터 변환"""
        print("🚀 2019-2024년 전체 데이터 UTF-8 변환 시작")
        print("=" * 60)
        
        years = [2019, 2020, 2021, 2022, 2023, 2024]
        converted_files = []
        
        for year in years:
            filepath = f'{year}_전국_감염병.csv'
            
            if Path(filepath).exists():
                result_df = self.convert_to_utf8_with_headers(filepath, year)
                if result_df is not None:
                    converted_files.append(f'{year}_감염병_UTF8.csv')
            else:
                print(f"❌ 파일 없음: {filepath}")
        
        print(f"\n🎉 변환 완료! 총 {len(converted_files)}개 파일")
        for filename in converted_files:
            print(f"  📁 data/kdca/{filename}")
        
        return converted_files
    
    def create_master_dataset(self):
        """통합 마스터 데이터셋 생성"""
        print("\n📊 통합 마스터 데이터셋 생성 중...")
        
        years = [2019, 2020, 2021, 2022, 2023, 2024]
        all_data = []
        
        for year in years:
            filepath = f'data/kdca/{year}_감염병_UTF8.csv'
            
            if Path(filepath).exists():
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                    # 연도 컬럼 추가
                    df.insert(0, '연도', year)
                    all_data.append(df)
                    print(f"✅ {year}년 데이터 로드: {df.shape}")
                except Exception as e:
                    print(f"❌ {year}년 로드 실패: {e}")
        
        if all_data:
            # 전체 데이터 통합
            master_df = pd.concat(all_data, ignore_index=True)
            
            # 마스터 파일 저장
            master_df.to_csv('data/kdca/kdca_master_dataset.csv', index=False, encoding='utf-8')
            
            print(f"🏆 마스터 데이터셋 생성 완료!")
            print(f"📊 크기: {master_df.shape}")
            print(f"📅 기간: 2019-2024년 ({len(years)}년)")
            print(f"🏙️ 지역: {len(master_df['지역'].unique())}개")
            
            # 미리보기
            print("\n📋 마스터 데이터 미리보기:")
            print(master_df[['연도', '지역', 'Infection_17', 'Infection_18', 'Infection_19']].head(10))
            
            return master_df
        else:
            print("❌ 통합할 데이터가 없습니다")
            return None
    
    def create_summary_report(self):
        """변환 결과 요약 보고서"""
        print("\n" + "=" * 60)
        print("🏆 질병관리청 데이터 UTF-8 변환 완료 보고서")
        print("=" * 60)
        
        # 변환된 파일들 확인
        converted_files = []
        years = [2019, 2020, 2021, 2022, 2023, 2024]
        
        for year in years:
            filepath = f'data/kdca/{year}_감염병_UTF8.csv'
            if Path(filepath).exists():
                file_size = Path(filepath).stat().st_size
                converted_files.append((year, filepath, file_size))
        
        print(f"📁 변환된 파일: {len(converted_files)}개")
        for year, filepath, size in converted_files:
            print(f"  {year}년: {filepath} ({size:,} bytes)")
        
        # 마스터 파일 확인
        master_path = 'data/kdca/kdca_master_dataset.csv'
        if Path(master_path).exists():
            master_size = Path(master_path).stat().st_size
            print(f"\n🎯 마스터 파일:")
            print(f"  📊 kdca_master_dataset.csv ({master_size:,} bytes)")
        
        print(f"\n✅ 다음 단계:")
        print(f"  - 데이터 정제 및 분석")
        print(f"  - 병무청 데이터와 통합")
        print(f"  - 팬데믹 영향도 분석")

def main():
    """메인 실행 함수"""
    converter = KDCAEncodingFixer()
    
    # 1. 개별 연도 변환
    converted_files = converter.convert_all_years()
    
    if converted_files:
        # 2. 마스터 데이터셋 생성
        master_df = converter.create_master_dataset()
        
        # 3. 요약 보고서
        converter.create_summary_report()
        
        return master_df
    else:
        print("❌ 변환 실패")
        return None

if __name__ == "__main__":
    main()