import os
from io import BytesIO

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from openai import OpenAI
import ollama  # 로컬 LLM용


# ==========================
# 0) 환경 변수 로딩 & LLM 제공자 결정
# ==========================
load_dotenv()

ENV_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 없을 수도 있음
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "gemma3")

# OPENAI 설정인데 키가 없으면 → 자동으로 ollama로 변경
if ENV_PROVIDER == "openai" and not OPENAI_API_KEY:
    EFFECTIVE_PROVIDER = "ollama"
    FALLBACK_REASON = "no_openai_key"
else:
    EFFECTIVE_PROVIDER = ENV_PROVIDER
    FALLBACK_REASON = None


class LLMClient:
    """OpenAI / Ollama를 하나의 인터페이스로 묶는 클라이언트"""

    def __init__(self, provider: str):
        self.provider = provider
        if provider == "openai":
            if not OPENAI_API_KEY:
                # 안전장치
                raise ValueError(
                    "OPENAI_API_KEY가 없습니다. .env에 키를 넣거나 LLM_PROVIDER=ollama로 변경하세요."
                )
            self.client = OpenAI()
        elif provider == "ollama":
            self.client = None  # ollama는 전역 함수로 호출
        else:
            raise ValueError(f"지원하지 않는 LLM provider: {provider}")

    # --------- 텍스트 요약 ----------
    def summarize(self, text: str, mode: str = "overview", level: str = "기본") -> str:
        """
        mode:
          - overview: 전체 개요 요약
          - keywords: 키워드/태그만 뽑기
          - toc: 문서 구조/목차 생성
          - highlights: 중요한 문장/포인트 추출
        level:
          - 짧게 / 기본 / 자세히 (overview 모드에서만 사용)
        """
        # -------- OpenAI 기반 요약 --------
        if self.provider == "openai":
            if mode == "overview":
                system_prompt = (
                    "당신은 한국어 문서를 분석하는 전문 비서입니다. "
                    "불필요한 수식어는 줄이고 핵심 아이디어 위주로 정리하세요."
                )
                if level == "짧게":
                    length_instruction = "한두 문장으로 아주 짧게 요약해줘."
                elif level == "자세히":
                    length_instruction = (
                        "핵심 주제, 주요 주장, 근거, 결론을 포함해서 15~20줄 정도로 자세히 요약해줘."
                    )
                else:  # 기본
                    length_instruction = (
                        "핵심 주제, 주요 주장, 중요한 인사이트를 포함해서 7~10줄 정도로 요약해줘."
                    )
                user_prompt = (
                    f"{length_instruction}\n\n"
                    f"다음은 요약할 문서 전체 내용이다:\n\n{text}"
                )
            elif mode == "keywords":
                system_prompt = (
                    "당신은 한국어 문서에서 핵심 키워드만 뽑아내는 전문가입니다. "
                    "중복되거나 의미가 약한 단어는 제외하세요."
                )
                user_prompt = (
                    "다음 문서에서 핵심 키워드/태그를 5~10개 정도 뽑아줘. "
                    "쉼표로 구분해서 한 줄로만 출력해줘.\n\n"
                    f"{text}"
                )
            elif mode == "toc":
                system_prompt = (
                    "당신은 한국어 기술 문서의 구조를 분석하여 논리적인 목차를 만드는 전문가입니다. "
                    "실제 문단 구조를 추론해 상위/하위 섹션을 정리하세요."
                )
                user_prompt = (
                    "다음 문서를 읽고, 실제 내용 순서에 맞는 목차를 만들어줘.\n"
                    "- 숫자 목록 형태로 출력해줘 (예: 1. 개요, 2. 배경, 3. 결론).\n"
                    "- 문서에 없는 내용을 새로 만들지 말고, 실제로 등장하는 주제만 사용해.\n\n"
                    f"{text}"
                )
            elif mode == "highlights":
                system_prompt = (
                    "당신은 긴 문서에서 중요한 문장만 골라 하이라이트해주는 전문가입니다. "
                    "핵심 주장, 중요한 정의, 결론에 해당하는 문장을 우선적으로 선택하세요."
                )
                user_prompt = (
                    "다음 문서에서 가장 중요한 문장 5~10개를 골라줘.\n"
                    "- 각 문장은 원문 그대로 발췌해.\n"
                    "- 새로운 문장을 만들지 말고, 문서에 실제로 있는 문장만 사용해.\n"
                    "- 각 문장 앞에는 '- '를 붙여 bullet 목록으로 출력해줘.\n\n"
                    f"{text}"
                )
            else:
                # 알려지지 않은 모드는 기본 요약으로 처리
                system_prompt = (
                    "당신은 한국어 문서를 분석하는 전문 비서입니다. "
                    "핵심 내용을 간결하게 요약해 주세요."
                )
                user_prompt = f"다음 문서를 요약해줘:\n\n{text}"

            resp = self.client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.output_text

        # -------- Ollama 기반 요약 --------
        elif self.provider == "ollama":
            if mode == "overview":
                system_prompt = (
                    "당신은 한국어 문서를 분석하는 오프라인 비서입니다. "
                    "수준 높게 요약하되, 불필요한 말은 줄이고 핵심만 정리하세요."
                )
                if level == "짧게":
                    length_instruction = "한두 문장으로 아주 짧게 요약해줘."
                elif level == "자세히":
                    length_instruction = (
                        "핵심 주제, 주요 주장, 근거, 결론을 포함해서 15~20줄 정도로 자세히 요약해줘."
                    )
                else:  # 기본
                    length_instruction = (
                        "핵심 주제, 주요 주장, 중요한 인사이트를 포함해서 7~10줄 정도로 요약해줘."
                    )
                user_content = (
                    f"{length_instruction}\n\n"
                    f"다음은 요약할 문서 전체 내용이다:\n\n{text}"
                )
            elif mode == "keywords":
                system_prompt = (
                    "당신은 한국어 문서에서 중요한 키워드만 추려내는 오프라인 비서입니다."
                )
                user_content = (
                    "다음 문서에서 핵심 키워드/태그를 5~10개 정도 뽑아줘. "
                    "쉼표로 구분해서 한 줄로만 출력해줘.\n\n"
                    f"{text}"
                )
            elif mode == "toc":
                system_prompt = (
                    "당신은 한국어 문서를 읽고 구조를 파악해 목차를 만들어주는 오프라인 비서입니다. "
                    "문서의 흐름을 보고 상위/하위 섹션을 자연스럽게 나누세요."
                )
                user_content = (
                    "다음 문서를 읽고, 실제 내용 순서에 맞는 목차를 만들어줘.\n"
                    "- 숫자 목록 형태로 출력해줘 (예: 1. 개요, 2. 배경, 3. 결론).\n"
                    "- 문서에 없는 내용을 새로 만들지 말고, 실제로 등장하는 주제만 사용해.\n\n"
                    f"{text}"
                )
            elif mode == "highlights":
                system_prompt = (
                    "당신은 긴 한국어 문서에서 중요 문장만 골라주는 오프라인 비서입니다. "
                    "핵심 주장, 정의, 결론을 대표하는 문장을 선택하세요."
                )
                user_content = (
                    "다음 문서에서 가장 중요한 문장 5~10개를 골라줘.\n"
                    "- 각 문장은 원문 그대로 발췌해.\n"
                    "- 새로운 문장을 만들지 말고, 문서에 실제로 있는 문장만 사용해.\n"
                    "- 각 문장 앞에는 '- '를 붙여 bullet 목록으로 출력해줘.\n\n"
                    f"{text}"
                )
            else:
                system_prompt = (
                    "당신은 한국어 문서를 분석하는 오프라인 비서입니다. "
                    "핵심 내용을 간결하게 요약해 주세요."
                )
                user_content = f"다음 문서를 요약해줘:\n\n{text}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
            # ollama-python에서 message는 dict일 수도 있고 객체일 수도 있어서 둘 다 처리
            if hasattr(response, "message"):
                msg = response.message
                if isinstance(msg, dict):
                    return msg.get("content", "")
                return msg.content
            return ""

        else:
            raise ValueError("지원하지 않는 provider")

    # --------- 임베딩 ----------
    def embed(self, texts: list[str]) -> np.ndarray:
        """여러 문장에 대해 임베딩 벡터 생성"""
        if self.provider == "openai":
            emb = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            vectors = [d.embedding for d in emb.data]
            return np.array(vectors, dtype="float32")

        elif self.provider == "ollama":
            resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=texts)
            return np.array(resp["embeddings"], dtype="float32")

        else:
            raise ValueError("지원하지 않는 provider")


# 전역 LLM 클라이언트 하나만 생성
llm = LLMClient(EFFECTIVE_PROVIDER)


# ==========================
# 2) 유틸 함수들
# ==========================
def extract_text_from_file(uploaded_file) -> str:
    """PDF / TXT에서 텍스트 추출 (이미지 OCR 포함)"""
    name_lower = uploaded_file.name.lower()

    # PDF
    if uploaded_file.type == "application/pdf" or name_lower.endswith(".pdf"):
        data = uploaded_file.read()
        reader = PdfReader(BytesIO(data))
        texts = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 10:
                # 텍스트가 있으면 그대로 사용
                texts.append(text)
            else:
                # 텍스트가 거의 없으면 OCR 시도
                st.write(f"📷 페이지 {page_num+1} OCR 분석 중...")
                images = convert_from_bytes(
                    data,
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    poppler_path=r"C:\poppler\Library\bin",
                )
                ocr_text = ""
                for img in images:
                    ocr_text += pytesseract.image_to_string(img, lang="kor+eng")
                texts.append(ocr_text)

        return "\n".join(texts)

    # TXT
    if uploaded_file.type.startswith("text/") or name_lower.endswith(".txt"):
        data = uploaded_file.read()
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("cp949", errors="ignore")

    return ""


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """긴 텍스트를 일정 길이로 나누기"""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, D), b: (D,) → (N,) 코사인 유사도"""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm)


# ==========================
# 3) 세션 상태 초기화
# ==========================
if "docs" not in st.session_state:
    st.session_state.docs = []  # 각 요소: dict(id, name, text, chunks, embeddings, overview, keywords, toc, highlights)

if "compare_result" not in st.session_state:
    st.session_state.compare_result = None

if "chat_histories" not in st.session_state:
    # 문서별 채팅 히스토리: {doc_id: [ {role: "user"/"assistant", content: "..."} ]}
    st.session_state.chat_histories = {}


# ==========================
# 1) 화면 제목 & 모드 안내
# ==========================
st.set_page_config(page_title="Synapse Docs", page_icon="📘", layout="wide")

st.title("📘 Synapse Docs")
st.caption("AI 문서 요약 & 맥락 분석 비서 (PDF / TXT 지원)")

if FALLBACK_REASON == "no_openai_key":
    st.toast("OPENAI_API_KEY가 없어 자동으로 Ollama 모드로 전환했습니다.", icon="⚙️")
else:
    st.toast(f"현재 LLM 제공자: {EFFECTIVE_PROVIDER}", icon="🤖")


# ==========================
# 4) 사이드바 – 문서 업로드 & 선택
# ==========================
with st.sidebar:
    st.header("📂 문서 업로드")

    uploaded_files = st.file_uploader(
        "PDF / TXT 파일 여러 개 업로드",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("문서 분석 시작", use_container_width=True) and uploaded_files:
        for f in uploaded_files:
            with st.spinner(f"▶ {f.name} 분석 중..."):
                raw_text = extract_text_from_file(f)
                if not raw_text.strip():
                    st.warning(f"{f.name}: 텍스트를 추출하지 못했습니다.")
                    continue

                chunks = chunk_text(raw_text, max_chars=1200)
                embeddings = llm.embed(chunks)

                # 개요 요약 & 키워드 추출 (긴 문서면 앞부분만 요약에 사용)
                base_for_summary = raw_text[:8000]
                overview = llm.summarize(base_for_summary, mode="overview", level="기본")
                keywords = llm.summarize(base_for_summary, mode="keywords")
                toc = llm.summarize(base_for_summary, mode="toc")
                highlights = llm.summarize(base_for_summary, mode="highlights")

                doc_id = len(st.session_state.docs)
                st.session_state.docs.append(
                    {
                        "id": doc_id,
                        "name": f.name,
                        "text": raw_text,
                        "chunks": chunks,
                        "embeddings": embeddings,
                        "overview": overview,
                        "keywords": keywords,
                        "toc": toc,
                        "highlights": highlights,
                    }
                )
        st.success("문서 분석 완료!")

    st.markdown("---")
    st.header("📑 문서 선택")

    if st.session_state.docs:
        doc_names = [d["name"] for d in st.session_state.docs]
        current_doc_name = st.selectbox(
            "분석할 문서를 선택하세요",
            options=doc_names,
            index=0,
            key="selected_doc_name",
        )
    else:
        current_doc_name = None


# ==========================
# 5) 메인 영역 – 선택 문서 상세
# ==========================
if not st.session_state.docs:
    st.info("왼쪽 사이드바에서 문서를 업로드하고 '문서 분석 시작'을 눌러주세요.")
else:
    # 선택된 문서 찾기
    selected_idx = next(
        i for i, d in enumerate(st.session_state.docs) if d["name"] == current_doc_name
    )
    selected_doc = st.session_state.docs[selected_idx]

    st.subheader(f"📄 선택된 문서: {selected_doc['name']}")

    col1, col2 = st.columns([2, 1])

    # ===== 요약 + 요약 수준 선택 =====
    with col1:
        st.markdown("#### 📌 문서 개요 요약")

        # 요약 수준 선택
        summary_level = st.radio(
            "요약 수준 선택",
            ["짧게", "기본", "자세히"],
            horizontal=True,
            key=f"summary_level_{selected_doc['id']}",
        )

        if st.button("이 수준으로 다시 요약", key=f"resummarize_{selected_doc['id']}"):
            with st.spinner("선택한 수준으로 다시 요약 중..."):
                base_for_summary = selected_doc["text"][:8000]
                new_overview = llm.summarize(
                    base_for_summary, mode="overview", level=summary_level
                )
                # 세션 상태의 문서 정보 업데이트
                st.session_state.docs[selected_idx]["overview"] = new_overview
                selected_doc["overview"] = new_overview
                st.success("요약을 새로 생성했습니다.")

        st.write(selected_doc["overview"])

    # ===== 키워드 + 요약 다운로드 =====
    with col2:
        st.markdown("#### 🏷️ 키워드 / 태그")
        st.write(selected_doc["keywords"])

        # 요약 다운로드 버튼
        download_text = (
            f"문서명: {selected_doc['name']}\n\n"
            f"[요약]\n{selected_doc['overview']}\n\n"
            f"[키워드]\n{selected_doc['keywords']}\n"
        )
        st.download_button(
            "💾 현재 요약을 TXT로 저장",
            data=download_text,
            file_name=f"{selected_doc['name']}_summary.txt",
            mime="text/plain",
            key=f"download_{selected_doc['id']}",
        )

    st.markdown("---")
    # 자동 생성 목차
    with st.expander("📚 자동 생성 목차", expanded=False):
        toc_text = selected_doc.get("toc", "").strip()
        if toc_text:
            st.markdown(toc_text)
        else:
            st.write("목차 정보가 아직 생성되지 않았습니다.")

    # 중요 문장 하이라이트
    with st.expander("✨ 중요 문장 하이라이트", expanded=False):
        hl_text = selected_doc.get("highlights", "").strip()
        if hl_text:
            st.markdown(hl_text)
        else:
            st.write("중요 문장 하이라이트가 아직 생성되지 않았습니다.")

    # 원문 미리보기
    with st.expander("📝 원문 텍스트 미리보기 (앞부분)", expanded=False):
        st.text(selected_doc["text"][:4000])

    # ===== 여러 문서 비교 / 통합 요약 =====
    if len(st.session_state.docs) >= 2:
        st.markdown("---")
        st.markdown("### 📊 여러 문서 비교 / 통합 요약")

        doc_names = [d["name"] for d in st.session_state.docs]
        default_compare = [selected_doc["name"]]

        selected_for_compare = st.multiselect(
            "비교할 문서를 선택하세요 (최소 2개)",
            options=doc_names,
            default=default_compare,
            key="compare_select",
        )

        if (
            len(selected_for_compare) >= 2
            and st.button("🔍 선택한 문서 비교 요약 생성", key="compare_button")
        ):
            with st.spinner("여러 문서를 비교 분석 중..."):
                blocks = []
                for d in st.session_state.docs:
                    if d["name"] in selected_for_compare:
                        blocks.append(
                            f"[문서명]: {d['name']}\n"
                            f"[요약]: {d['overview']}\n"
                            f"[키워드]: {d['keywords']}\n"
                            f"[본문 앞부분]: {d['text'][:1500]}\n"
                        )
                compare_input = "\n\n====================\n\n".join(blocks)

                if EFFECTIVE_PROVIDER == "openai":
                    system_prompt = (
                        "여러 개의 한국어 문서를 비교 분석하는 전문가입니다. "
                        "각 문서의 공통점, 차이점, 특징적인 부분을 정리해 주세요."
                    )
                    user_prompt = (
                        "아래에 여러 문서의 요약·키워드·본문 일부가 정리되어 있다.\n"
                        "- 각 문서의 공통점과 차이점을 항목별로 정리해줘.\n"
                        "- 어떤 문서가 어떤 관점/주제를 더 강조하는지도 설명해줘.\n"
                        "- 마지막에는 종합적인 결론을 3~5줄로 정리해줘.\n\n"
                        f"{compare_input}"
                    )
                    resp = llm.client.responses.create(
                        model="gpt-4.1-mini",
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    st.session_state.compare_result = resp.output_text
                else:
                    system_prompt = (
                        "당신은 여러 개의 한국어 문서를 비교 분석하는 오프라인 비서입니다. "
                        "공통점과 차이점을 구조적으로 정리하세요."
                    )
                    user_content = (
                        "아래에 여러 문서의 요약·키워드·본문 일부가 정리되어 있다.\n"
                        "- 각 문서의 공통점과 차이점을 항목별로 정리해줘.\n"
                        "- 어떤 문서가 어떤 관점/주제를 더 강조하는지도 설명해줘.\n"
                        "- 마지막에는 종합적인 결론을 3~5줄로 정리해줘.\n\n"
                        f"{compare_input}"
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ]
                    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
                    if hasattr(response, "message"):
                        msg = response.message
                        if isinstance(msg, dict):
                            st.session_state.compare_result = msg.get("content", "")
                        else:
                            st.session_state.compare_result = msg.content
                    else:
                        st.session_state.compare_result = "비교 결과를 생성하지 못했습니다."

        if st.session_state.compare_result:
            st.markdown("#### 비교 요약 결과")
            st.write(st.session_state.compare_result)

    # ===== Q&A 섹션 =====
    st.markdown("---")
    st.header("💬 문서 기반 Q&A")

    # 이 문서에 대한 개별 히스토리 가져오기
    doc_id = selected_doc["id"]
    chat_history = st.session_state.chat_histories.setdefault(doc_id, [])

    # 기존 대화 출력
    for msg in chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    # 사용자 질문 입력
    user_question = st.chat_input(
        "문서 내용을 기반으로 질문을 입력하세요. (문서에 없는 내용은 답변하지 않도록 설정됨)"
    )

    if user_question:
        # 1) 히스토리에 사용자 질문 추가 & 화면 표시
        chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        # 2) 임베딩 기반 관련 조각 찾기
        question_vec = llm.embed([user_question])[0]
        sims = cosine_sim_matrix(selected_doc["embeddings"], question_vec)
        top_k = 5
        top_idx = np.argsort(-sims)[:top_k]
        context_blocks = [selected_doc["chunks"][i] for i in top_idx]

        context_text = "\n\n---\n\n".join(context_blocks)

        # 3) 시스템 프롬프트 – 문서에 없는 내용은 무조건 "없다"로 답하게
        system_instruction = (
            "너는 로컬에서 실행되는 한국어 전용 오프라인 문서 비서야. "
            "반드시 아래 규칙을 지켜야 한다.\n"
            "1) 답변에 포함되는 모든 사실, 인물, 기관, 날짜, 수치는 "
            "[문서 컨텍스트] 안에 실제로 등장하는 내용에서만 가져와라.\n"
            "2) [문서 컨텍스트]에 등장하지 않는 인물·기관·사실에 대해 "
            "사용자가 질문하면 반드시 아래 중 하나로만 답해라:\n"
            "   - '이 문서에는 그 인물(정보)에 대한 내용이 없습니다.'\n"
            "   - '이 문서에서는 해당 주제나 인물이 언급되지 않습니다.'\n"
            "3) 절대 상식, 추측, 학습된 지식을 이용해 내용을 보완하거나 만들어내지 마라.\n"
            "4) 문서에 있는 내용만 근거로 요약하거나 인용하라.\n"
            "5) 답변은 항상 한국어로만 작성하고, 영어·중국어 등 다른 언어를 사용하지 마라.\n"
            "6) 문서에 근거 없는 내용으로 대답하려고 시도할 경우, 즉시 중단하고 위 문장을 그대로 출력해라."
        )

        # 4) LLM 호출
        if EFFECTIVE_PROVIDER == "openai":
            resp = llm.client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": system_instruction},
                    {
                        "role": "user",
                        "content": (
                            "다음은 이 문서에서 추출한 일부 컨텍스트이다.\n\n"
                            f"[문서 컨텍스트]\n{context_text}\n\n"
                            f"[사용자 질문]\n{user_question}"
                        ),
                    },
                ],
            )
            answer = resp.output_text
        else:
            messages = [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": (
                        "다음은 이 문서에서 추출한 일부 컨텍스트이다.\n\n"
                        f"[문서 컨텍스트]\n{context_text}\n\n"
                        f"[사용자 질문]\n{user_question}"
                    ),
                },
            ]
            response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
            if hasattr(response, "message"):
                msg = response.message
                if isinstance(msg, dict):
                    answer = msg.get("content", "")
                else:
                    answer = msg.content
            else:
                answer = "답변을 생성하지 못했습니다."

        # 5) 답변을 히스토리에 추가 + 화면에 표시
        chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

        # 6) 이번 답변에서 참고한 문서 조각 보여주기
        with st.expander("📎 이번 답변에서 참고한 문서 조각 보기", expanded=False):
            for i, idx in enumerate(top_idx, start=1):
                st.markdown(f"**조각 {i} (유사도: {sims[idx]:.3f})**")
                st.text(selected_doc["chunks"][idx])
                st.markdown("---")

