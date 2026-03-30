from __future__ import annotations

import json
import streamlit as st

from gemini_utils import (
    DEFAULT_MODEL,
    SUPPORTED_FILE_TYPES,
    SUPPORTED_TARGET_LANGUAGES,
    build_docx_bytes,
    build_json_bytes,
    build_translation_context,
    call_gemini_for_translation,
    chat_with_context,
    file_to_bytes,
    render_diagram_explanations,
)
from history_utils import clear_history, list_history_records, load_history_record, save_history_record


st.set_page_config(page_title="Gemini Translator", layout="wide")
st.title("Gemini Translator V3")
st.caption("Fast mode + deep scan + history + chat + visual crop explanation")

if "translator_outputs" not in st.session_state:
    st.session_state.translator_outputs = {}

if "translator_chat_history" not in st.session_state:
    st.session_state.translator_chat_history = {}

with st.sidebar:
    st.header("Translator Settings")
    target_language = st.selectbox("Translate into", SUPPORTED_TARGET_LANGUAGES, index=0)
    model_name = st.text_input("Gemini model", value=DEFAULT_MODEL)
    scan_mode = st.radio("Scan mode", ["Fast mode", "Deep scan"], index=0)
    deep_scan = scan_mode == "Deep scan"
    auto_save = st.checkbox("Auto-save history", value=True)

    st.markdown("---")
    st.subheader("Saved translator history")
    records = list_history_records(prefix="translator_")

    if records:
        chosen = st.selectbox("Choose saved record", [""] + [r.name for r in records], index=0)
        if chosen:
            record = load_history_record(next(r for r in records if r.name == chosen))
            st.download_button(
                "Download selected history JSON",
                data=json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=chosen,
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.info("No saved translator history.")

    if st.button("Clear translator history", use_container_width=True):
        clear_history(prefix="translator_")
        st.success("Translator history cleared.")
        st.rerun()

uploaded_files = st.file_uploader(
    "Upload PDF, image, TXT, or DOCX files",
    type=SUPPORTED_FILE_TYPES,
    accept_multiple_files=True,
)

if st.button("Translate files", use_container_width=True):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        progress = st.progress(0)
        total = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files, start=1):
            file_bytes = file_to_bytes(uploaded_file)
            with st.spinner(f"Processing {uploaded_file.name}..."):
                output = call_gemini_for_translation(
                    filename=uploaded_file.name,
                    file_bytes=file_bytes,
                    target_language=target_language,
                    model_name=model_name.strip() or DEFAULT_MODEL,
                    deep_scan=deep_scan,
                )

            st.session_state.translator_outputs[uploaded_file.name] = output
            st.session_state.translator_chat_history.setdefault(uploaded_file.name, [])

            if auto_save:
                payload = json.loads(build_json_bytes(output).decode("utf-8"))
                save_history_record("translator", uploaded_file.name, payload)

            progress.progress(i / total)

for file_name, output in st.session_state.translator_outputs.items():
    st.markdown("---")
    st.header(file_name)

    st.subheader("Detected language")
    st.write(output.detected_language)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original extracted text")
        st.text_area(
            f"Original {file_name}",
            value=output.original_text,
            height=320,
            key=f"translator_orig_{file_name}",
        )

    with c2:
        st.subheader(f"Direct translation ({output.target_language})")
        st.text_area(
            f"Translated {file_name}",
            value=output.translated_text,
            height=320,
            key=f"translator_trans_{file_name}",
        )

    st.subheader("2–3 simple examples in layman English")
    if output.examples:
        for n, ex in enumerate(output.examples, start=1):
            st.markdown(f"{n}. {ex}")
    else:
        st.info("No examples returned.")

    render_diagram_explanations(output.diagrams)

    payload = json.loads(build_json_bytes(output).decode("utf-8"))

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            f"Download JSON - {file_name}",
            data=build_json_bytes(output),
            file_name=f"{file_name}_translation.json",
            mime="application/json",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            f"Download DOCX - {file_name}",
            data=build_docx_bytes(file_name, output),
            file_name=f"{file_name}_translation.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    with d3:
        if st.button(f"Save result - {file_name}", key=f"save_{file_name}"):
            save_path = save_history_record("translator", file_name, payload)
            st.success(f"Saved: {save_path.name}")

    st.subheader("Chat with this file")
    history = st.session_state.translator_chat_history[file_name]
    for msg in history:
        st.markdown(f"**{'You' if msg['role'] == 'user' else 'Assistant'}:** {msg['content']}")

    prompt = st.text_input(f"Ask about {file_name}", key=f"chat_{file_name}")
    if st.button(f"Send - {file_name}", key=f"send_{file_name}"):
        if prompt.strip():
            history.append({"role": "user", "content": prompt})
            context = build_translation_context(output)
            answer = chat_with_context(context, prompt, model_name=model_name.strip() or DEFAULT_MODEL)
            history.append({"role": "assistant", "content": answer})
            save_history_record("translator_chat", file_name, {"file_name": file_name, "chat_history": history})
            st.rerun()
