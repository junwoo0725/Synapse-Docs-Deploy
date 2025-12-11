# Synapse Docs – AI 문서 요약 · 비교 · 하이라이트 시스템

로컬 LLM(Ollama)과 OCR(Tesseract + Poppler)을 활용해
PDF/TXT 문서를 요약·비교·목차·하이라이트·Q&A까지 지원하는 문서 분석 서비스입니다.

## 개발 환경

- Python 3.11.9
- Streamlit
- Ollama (gemma2:2b, nomic-embed-text)
- Poppler for Windows
- Tesseract OCR

## 설치 방법
1. Python 3.11.9
공식 사이트: https://www.python.org

 2.Ollama + 모델
Ollama 설치: https://ollama.com/download gemma3:4b

 3.Poppler for Windows
PDF → 이미지 변환용 (pdf2image에서 사용)
깃헙 릴리즈: https://github.com/oschwartz10612/poppler-windows/releases/

 4.Tesseract OCR
이미지(PDF 스캔본) → 텍스트 인식용
UB Mannheim 빌드: https://github.com/UB-Mannheim/tesseract/wiki

```bash
# 1. 가상환경 생성 및 활성화 (Windows)
python -m venv venv
venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt
