import io
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from docx import Document
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

try:
    import cv2
except ImportError:
    st.error("❌ OpenCV Error: cv2 module not found.")
    st.error("🔧 Fix: Add 'opencv-python-headless' to requirements.txt and redeploy.")
    st.stop()

try:
    import fitz
except ImportError:
    st.error("❌ PyMuPDF Error: fitz module not found.")
    st.error("🔧 Fix: Add 'PyMuPDF' to requirements.txt and redeploy.")
    st.stop()

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Babuchiti", page_icon="📘", layout="wide")

MODEL_NAME = "gemini-2.5-flash"
HISTORY_FILE = "history.json"
LOGO_FILE = "ABT LOGO.jpg"

# =========================
# STYLES
# =========================
st.markdown(
    """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}
.sidebar-logo {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 18px;
}
.main-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
}
.chat-shell {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
}
.hero-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 4px;
}
.hero-sub {
    text-align: center;
    color: #4b5563;
    margin-bottom: 20px;
}
.chat-wrap {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 16px;
    min-height: 420px;
    max-height: 520px;
    overflow-y: auto;
}
.user-msg {
    background: #111827;
    color: white;
    padding: 12px 14px;
    border-radius: 16px;
    margin: 10px 0 10px auto;
    width: fit-content;
    max-width: 80%;
    white-space: pre-wrap;
}
.assistant-msg {
    background: #e5e7eb;
    color: #111827;
    padding: 12px 14px;
    border-radius: 16px;
    margin: 10px auto 10px 0;
    width: fit-content;
    max-width: 80%;
    white-space: pre-wrap;
}
.small-note {
    color: #6b7280;
    font-size: 13px;
}
.result-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 16px;
}
.hist-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
}
.hist-title {
    font-weight: 700;
    margin-bottom: 6px;
}
.crop-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# SESSION STATE
# =========================
defaults = {
    "current_view": "Home",
    "output_language": "English",
    "chat_messages": [],
    "translate_original": "",
    "translate_translated": "",
    "ocr_result": "",
    "analysis_result": "",
    "translate_crops": [],
    "ocr_crops": [],
    "analysis_crops": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# CLIENT
# =========================
def get_gemini_api_key() -> str:
    return (
        st.secrets.get("GEMINI_API_KEY", "")
        or os.getenv("GEMINI_API_KEY", "")
    ).strip()


def get_client():
    gemini_api_key = get_gemini_api_key()
    if not gemini_api_key:
        st.error("❌ API Key Error: GEMINI_API_KEY was not found.")
        st.error("🔧 Fix: Add GEMINI_API_KEY to Streamlit secrets or your .env file.")
        st.stop()
    if gemini_api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        st.error("❌ API Key Error: Please replace the placeholder API key.")
        st.stop()
    return genai.Client(api_key=gemini_api_key)

# =========================
# HISTORY
# =========================
def load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(items: List[Dict[str, Any]]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def add_history(entry_type: str, title: str, payload: Dict[str, Any]) -> None:
    items = load_history()
    items.insert(0, {"type": entry_type, "title": title, "payload": payload})
    save_history(items[:100])

# =========================
# BASIC FILE HELPERS
# =========================
def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return bio.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")


def read_docx(file_bytes: bytes) -> str:
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_pdf_if_possible(file_bytes: bytes) -> str:
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    parts = []
    for page in pdf:
        text = page.get_text("text")
        if text.strip():
            parts.append(text.strip())
    pdf.close()
    return "\n\n".join(parts).strip()


def render_pdf_pages(file_bytes: bytes, zoom: float = 2.0) -> List[Image.Image]:
    images: List[Image.Image] = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    for page in pdf:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    pdf.close()
    return images

# =========================
# CROP DETECTION
# =========================
def detect_diagram_crops(
    pil_img: Image.Image,
    min_w_ratio: float = 0.15,
    min_h_ratio: float = 0.15,
    max_crops: int = 8,
) -> List[Image.Image]:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    min_w = int(w * min_w_ratio)
    min_h = int(h * min_h_ratio)

    boxes: List[Tuple[int, int, int, int, int]] = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if bw < min_w or bh < min_h:
            continue
        if bw > int(w * 0.90) and bh > int(h * 0.90):
            continue

        aspect = bw / max(bh, 1)
        if aspect > 6 and bh < int(h * 0.15):
            continue
        if area < (w * h) * 0.02:
            continue
        if aspect < 4 or aspect > 8:
            boxes.append((x, y, bw, bh, area))

    boxes.sort(key=lambda b: b[4], reverse=True)

    merged: List[Tuple[int, int, int, int, int]] = []
    for box in boxes:
        x, y, bw, bh, area = box
        added = False
        for i, mb in enumerate(merged):
            mx, my, mw, mh, _ = mb
            if not (x > mx + mw or mx > x + bw or y > my + mh or my > y + bh):
                nx = min(x, mx)
                ny = min(y, my)
                nr = max(x + bw, mx + mw)
                nb = max(y + bh, my + mh)
                merged[i] = (nx, ny, nr - nx, nb - ny, (nr - nx) * (nb - ny))
                added = True
                break
        if not added:
            merged.append((x, y, bw, bh, area))

    merged.sort(key=lambda b: b[4], reverse=True)
    merged = merged[:max_crops]

    crops: List[Image.Image] = []
    for x, y, bw, bh, _ in merged:
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        crop = pil_img.crop((x1, y1, x2, y2))
        crops.append(crop)

    return crops

# =========================
# PROMPTS
# =========================
def build_ocr_prompt() -> str:
    return """
You are an exact OCR engine.

Rules:
- Extract the text exactly as it appears.
- Preserve line breaks where reasonably possible.
- Do not summarize.
- Do not explain.
- Do not add missing words.
- If the text is modern, classical, archaic, stylized, mixed-language, or scanned, still extract faithfully.
- Return only the extracted text.
""".strip()


def build_translation_prompt(target_language: str) -> str:
    return f"""
You are a strict translator.

Rules:
- Detect the source language automatically.
- Translate the text into {target_language}.
- Do not summarize.
- Do not omit content.
- Do not add explanation inside the translation.
- Preserve the meaning closely.
- If the text is classical, archaic, historical, or liturgical, still translate carefully into clear {target_language}.
- Return only the translated text.
""".strip()


def build_analysis_prompt(question: str) -> str:
    return f"""
You are a document analyzer.

Rules:
- Answer only from the uploaded document.
- If the document does not contain the answer, say exactly:
"The document does not provide that information."
- Be direct and clear.

Question:
{question}
""".strip()


def build_chat_prompt(message: str, target_language: str, document_text: str) -> str:
    base = f"""
You are Babuchiti AI.

Rules:
- Be direct, clear, and helpful.
- If a document is attached, use the attached document carefully.
- The user can ask normal questions, document analysis questions, or translation questions in this chat area.
- If the user asks to translate, detect the source language automatically and translate into {target_language}.
- If the user asks about a document, answer from the document.
- If the document does not contain the answer, say:
"The document does not provide that information."
- Do not hallucinate.

User message:
{message}
""".strip()

    if document_text.strip():
        base += f"\n\nATTACHED DOCUMENT CONTENT:\n{document_text}"
    return base


def build_crop_explain_prompt() -> str:
    return """
Describe this cropped visual region directly.

Rules:
- Say what is visibly present.
- If it looks like a talisman, symbol, seal, diagram, table, figure, or illustration, say so.
- Do not invent hidden meaning.
- Keep it short and direct.
""".strip()

# =========================
# GEMINI CALLS
# =========================
def call_gemini_text(prompt: str) -> str:
    try:
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return getattr(response, "text", "") or ""
    except Exception as e:
        err = str(e)
        if "getaddrinfo failed" in err or "11002" in err:
            return "Network error: Unable to connect to Gemini API. Please check your internet connection and try again."
        if "API_KEY" in err or "401" in err or "403" in err:
            return "API key error: Please check your Gemini API key."
        if "429" in err:
            return "Rate limit error: Too many requests. Please try again in a moment."
        if any(code in err for code in ["500", "502", "503"]):
            return "Server error: Gemini API is temporarily unavailable. Please try again later."
        return f"Error: {err}"


def call_gemini_with_images(prompt: str, images: List[Image.Image], mime_type: str = "image/png") -> str:
    try:
        client = get_client()
        contents: List[Any] = [prompt]
        for img in images:
            contents.append(types.Part.from_bytes(data=pil_to_bytes(img, "PNG"), mime_type=mime_type))

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return getattr(response, "text", "") or ""
    except Exception as e:
        err = str(e)
        if "getaddrinfo failed" in err or "11002" in err:
            return "Network error: Unable to connect to Gemini API. Please check your internet connection and try again."
        if "API_KEY" in err or "401" in err or "403" in err:
            return "API key error: Please check your Gemini API key."
        if "429" in err:
            return "Rate limit error: Too many requests. Please try again in a moment."
        if any(code in err for code in ["500", "502", "503"]):
            return "Server error: Gemini API is temporarily unavailable. Please try again later."
        return f"Error: {err}"


def explain_crop(crop_img: Image.Image) -> str:
    return call_gemini_with_images(build_crop_explain_prompt(), [crop_img], "image/png")

# =========================
# CONTENT EXTRACTION
# =========================
def extract_content(uploaded_file) -> Dict[str, Any]:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    result: Dict[str, Any] = {
        "filename": filename,
        "original_text": "",
        "used_ocr": False,
        "crops": [],
    }

    if ext == "txt":
        result["original_text"] = read_txt(file_bytes)
        return result

    if ext == "docx":
        result["original_text"] = read_docx(file_bytes)
        return result

    if ext == "pdf":
        pages = render_pdf_pages(file_bytes)
        all_crops: List[Image.Image] = []
        for page_img in pages:
            all_crops.extend(detect_diagram_crops(page_img))

        direct_text = extract_text_from_pdf_if_possible(file_bytes)
        if direct_text.strip():
            result["original_text"] = direct_text
            result["crops"] = all_crops
            return result

        result["used_ocr"] = True
        result["original_text"] = call_gemini_with_images(build_ocr_prompt(), pages, "image/png")
        result["crops"] = all_crops
        return result

    if ext in ["png", "jpg", "jpeg", "webp"]:
        img = bytes_to_pil(file_bytes)
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        result["used_ocr"] = True
        result["original_text"] = call_gemini_with_images(build_ocr_prompt(), [img], mime)
        result["crops"] = detect_diagram_crops(img)
        return result

    raise ValueError("Unsupported file type. Use pdf, docx, txt, png, jpg, jpeg, or webp.")

# =========================
# RENDER HELPERS
# =========================
def render_history_cards(items: List[Dict[str, Any]], empty_text: str):
    if not items:
        st.markdown(f'<div class="small-note">{empty_text}</div>', unsafe_allow_html=True)
        return

    for item in items:
        payload = item.get("payload", {})
        st.markdown('<div class="hist-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="hist-title">{item.get("title", "Untitled")}</div>', unsafe_allow_html=True)
        st.caption(f"Type: {item.get('type', '')}")

        if item.get("type") == "translate":
            st.text(f"File: {payload.get('filename', '')}")
            st.text_area(
                "Original",
                value=payload.get("original_text", "")[:1200],
                height=120,
                disabled=True,
                key=f"hist_t_o_{id(item)}",
            )
            st.text_area(
                "Translated",
                value=payload.get("translated_text", "")[:1200],
                height=120,
                disabled=True,
                key=f"hist_t_t_{id(item)}",
            )
        elif item.get("type") == "ocr":
            st.text(f"File: {payload.get('filename', '')}")
            st.text_area(
                "OCR",
                value=payload.get("original_text", "")[:1600],
                height=140,
                disabled=True,
                key=f"hist_o_{id(item)}",
            )
        elif item.get("type") == "analysis":
            st.text(f"File: {payload.get('filename', '')}")
            st.text(f"Question: {payload.get('question', '')}")
            st.text_area(
                "Answer",
                value=payload.get("answer", "")[:1200],
                height=120,
                disabled=True,
                key=f"hist_a_{id(item)}",
            )
        else:
            st.text(f"File: {payload.get('filename', '')}")
            st.text(f"Message: {payload.get('message', '')}")
            st.text_area(
                "Answer",
                value=payload.get("answer", "")[:1200],
                height=120,
                disabled=True,
                key=f"hist_c_{id(item)}",
            )

        st.markdown("</div>", unsafe_allow_html=True)


def render_crops(crops: List[Image.Image], key_prefix: str):
    if not crops:
        st.caption("No cropped diagram/talisman detected.")
        return

    st.subheader("Detected Cropped Diagrams / Symbols")
    for i, crop in enumerate(crops, start=1):
        st.markdown('<div class="crop-box">', unsafe_allow_html=True)
        st.image(crop, caption=f"Crop {i}", use_container_width=True)
        if st.button(f"Explain Crop {i}", key=f"{key_prefix}_explain_{i}"):
            with st.spinner("Explaining crop..."):
                try:
                    explanation = explain_crop(crop)
                    st.text_area(
                        f"Crop {i} Explanation",
                        value=explanation,
                        height=120,
                        disabled=True,
                        key=f"{key_prefix}_explain_box_{i}",
                    )
                except Exception as e:
                    st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">BABUCHITI</div>', unsafe_allow_html=True)

    if st.button("Home", use_container_width=True):
        st.session_state.current_view = "Home"
    if st.button("Search Chats", use_container_width=True):
        st.session_state.current_view = "Search Chats"
    if st.button("DOCUMENT TRANSLATOR", use_container_width=True):
        st.session_state.current_view = "Document Translator"
    if st.button("OCR EDITOR", use_container_width=True):
        st.session_state.current_view = "OCR Editor"
    if st.button("DOCUMENT ANALYZER", use_container_width=True):
        st.session_state.current_view = "Document Analyzer"
    if st.button("OUTPUT LANGUAGE", use_container_width=True):
        st.session_state.current_view = "Output Language"
    if st.button("HISTORY", use_container_width=True):
        st.session_state.current_view = "History"

view = st.session_state.current_view

# =========================
# HOME / CHAT
# =========================
if view == "Home":
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size: 48px;">📘 BABUCHITI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size: 24px;">What is On Your Mind Today?</div>', unsafe_allow_html=True)

    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=300)
    else:
        st.caption("Logo file not found: ABT LOGO.jpg")

    if st.session_state.chat_messages:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state.chat_messages:
            cls = "user-msg" if msg["role"] == "user" else "assistant-msg"
            st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    chat_file = st.file_uploader(
        "",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="chat_file_uploader",
    )

    chat_message = st.text_area(
        "Ask Me Anything",
        placeholder="Ask Me Anything",
        height=90,
        key="chat_input_box",
        label_visibility="visible",
    )

    send_chat = st.button("Send", use_container_width=True)

    if chat_file is not None:
        st.markdown(f'<div class="small-note">Attached: {chat_file.name}</div>', unsafe_allow_html=True)

    if send_chat:
        if not chat_message.strip() and chat_file is None:
            st.warning("Type a message or attach a file.")
        else:
            user_text = chat_message.strip() or ""
            if chat_file is not None:
                user_text = (user_text + f"\n[Attached file: {chat_file.name}]").strip()

            st.session_state.chat_messages.append({"role": "user", "content": user_text})

            with st.spinner("Processing..."):
                try:
                    document_text = ""
                    filename = None
                    used_ocr = False

                    if chat_file is not None:
                        extracted = extract_content(chat_file)
                        document_text = extracted["original_text"].strip()
                        filename = extracted["filename"]
                        used_ocr = extracted["used_ocr"]

                    answer = call_gemini_text(
                        build_chat_prompt(
                            chat_message.strip() or "Please work with the attached document.",
                            st.session_state.output_language,
                            document_text,
                        )
                    )

                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

                    add_history(
                        "chat",
                        "Chat",
                        {
                            "filename": filename,
                            "message": chat_message.strip(),
                            "answer": answer,
                            "original_text": document_text[:5000],
                            "target_language": st.session_state.output_language,
                            "used_ocr": used_ocr,
                        },
                    )
                    st.rerun()

                except Exception as e:
                    st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# SEARCH CHATS
# =========================
elif view == "Search Chats":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Search Chats")
    st.write("Search through your saved chats and document history.")

    s1, s2 = st.columns([6, 1.5])
    with s1:
        search_query = st.text_input("Search history", placeholder="Search history", label_visibility="collapsed")
    with s2:
        do_search = st.button("Search", use_container_width=True)

    items = load_history()
    if do_search:
        q = search_query.lower().strip()
        if q:
            filtered = []
            for item in items:
                title = item.get("title", "").lower()
                payload = item.get("payload", {})
                haystack = " ".join(
                    [
                        payload.get("filename", ""),
                        payload.get("message", ""),
                        payload.get("question", ""),
                        payload.get("original_text", "")[:3000],
                        payload.get("translated_text", "")[:3000],
                        payload.get("answer", "")[:3000],
                    ]
                ).lower()
                if q in title or q in haystack:
                    filtered.append(item)
            items = filtered

    render_history_cards(items if do_search else [], "No search yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DOCUMENT TRANSLATOR
# =========================
elif view == "Document Translator":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("🌍 Document Translator")
    st.write("Translate your document. Source language is detected automatically.")

    trans_tab = st.radio(
        "Toggle",
        options=["Original", "Translated"],
        horizontal=True,
        key="translator_toggle",
        label_visibility="collapsed",
    )

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if trans_tab == "Original":
        st.text_area("Original Text", value=st.session_state.translate_original, height=400, disabled=True)
    else:
        st.text_area("Translated Text", value=st.session_state.translate_translated, height=400, disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        translate_file = st.file_uploader(
            "",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            key="translate_file_uploader",
        )
    with col2:
        do_translate = st.button("Translate", use_container_width=True)

    if do_translate:
        if translate_file is None:
            st.warning("Please upload a file.")
        else:
            with st.spinner("Translating..."):
                try:
                    extracted = extract_content(translate_file)
                    original_text = extracted["original_text"].strip()

                    if not original_text:
                        st.error("No text could be extracted from the file.")
                    else:
                        translated_text = call_gemini_text(
                            f"{build_translation_prompt(st.session_state.output_language)}\n\nSOURCE TEXT:\n{original_text}"
                        )

                        st.session_state.translate_original = original_text
                        st.session_state.translate_translated = translated_text
                        st.session_state.translate_crops = extracted["crops"]

                        add_history(
                            "translate",
                            f"Translate - {extracted['filename']}",
                            {
                                "filename": extracted["filename"],
                                "original_text": original_text[:5000],
                                "translated_text": translated_text[:5000],
                                "target_language": st.session_state.output_language,
                                "used_ocr": extracted["used_ocr"],
                            },
                        )
                        st.rerun()
                except Exception as e:
                    st.error(str(e))

    render_crops(st.session_state.translate_crops, "translator")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# OCR EDITOR
# =========================
elif view == "OCR Editor":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("OCR Editor")
    st.write("Extract the exact text of your document.")

    o1, o2 = st.columns([6, 1.5])
    with o1:
        ocr_file = st.file_uploader(
            "OCR File",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
            key="ocr_file_uploader",
            label_visibility="collapsed",
        )
    with o2:
        do_ocr = st.button("Extract Text", use_container_width=True)

    if ocr_file is not None:
        st.info(f"File selected: {ocr_file.name}")
    else:
        st.info("No file selected")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.text_area("Extracted Text", value=st.session_state.ocr_result, height=400, disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if do_ocr:
        if ocr_file is None:
            st.warning("Please upload a file.")
        else:
            with st.spinner("Extracting..."):
                try:
                    extracted = extract_content(ocr_file)
                    st.session_state.ocr_result = extracted["original_text"]
                    st.session_state.ocr_crops = extracted["crops"]

                    add_history(
                        "ocr",
                        f"OCR - {extracted['filename']}",
                        {
                            "filename": extracted["filename"],
                            "original_text": extracted["original_text"],
                            "used_ocr": extracted["used_ocr"],
                        },
                    )
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    render_crops(st.session_state.ocr_crops, "ocr")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DOCUMENT ANALYZER
# =========================
elif view == "Document Analyzer":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📊 Document Analyzer")
    st.write("Upload a document and ask a question about it.")

    analyze_file = st.file_uploader(
        "",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
        key="analyze_file_uploader",
        label_visibility="collapsed",
    )

    analysis_question = st.text_area(
        "Question",
        placeholder="Ask any question about the uploaded document",
        height=120,
        key="analyze_question_box",
        label_visibility="collapsed",
    )

    if st.button("Analyze", use_container_width=False):
        if analyze_file is None:
            st.warning("Please upload a file.")
        elif not analysis_question.strip():
            st.warning("Please ask a question.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    extracted = extract_content(analyze_file)
                    original_text = extracted["original_text"].strip()

                    if not original_text:
                        st.error("No text could be extracted from the file.")
                    else:
                        answer = call_gemini_text(
                            f"{build_analysis_prompt(analysis_question.strip())}\n\nDOCUMENT CONTENT:\n{original_text}"
                        )
                        st.session_state.analysis_result = answer
                        st.session_state.analysis_crops = extracted["crops"]

                        add_history(
                            "analysis",
                            f"Analyze - {extracted['filename']}",
                            {
                                "filename": extracted["filename"],
                                "question": analysis_question.strip(),
                                "answer": answer,
                                "original_text": original_text[:5000],
                                "used_ocr": extracted["used_ocr"],
                            },
                        )
                        st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.text_area("Answer", value=st.session_state.analysis_result, height=350, disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

    render_crops(st.session_state.analysis_crops, "analysis")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# OUTPUT LANGUAGE
# =========================
elif view == "Output Language":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Output Language")
    st.write("Select the language the translator will translate into.")

    options = ["English", "French", "Arabic", "Yoruba", "Hausa", "Igbo"]
    lang = st.selectbox(
        "Language",
        options,
        index=options.index(st.session_state.output_language),
        key="language_select_box",
    )

    if st.button("Save", use_container_width=False):
        st.session_state.output_language = lang
        st.success(f"Current output language: {lang}")

    st.info(f"Current output language: {st.session_state.output_language}")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# HISTORY
# =========================
elif view == "History":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([5, 1.5])
    with c1:
        st.subheader("History")
        st.write("All chats and documents you worked on are stored here.")
    with c2:
        if st.button("Clear History", use_container_width=True):
            save_history([])
            st.success("History cleared.")
            st.rerun()

    render_history_cards(load_history(), "No history yet.")
    st.markdown("</div>", unsafe_allow_html=True)
