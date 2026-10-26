import html

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="NLP Toolkit", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
        }
        h1, h2, h3, p, label, .stMarkdown {
            color: #e2e8f0 !important;
        }
        div[data-testid="stTextArea"] textarea {
            background-color: #1e293b;
            color: #f1f5f9;
            border: 1px solid #334155;
        }
        .result-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.25rem;
            margin-top: 1rem;
        }
        .stButton button {
            background-color: #6366f1;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1.5rem;
        }
        .stButton button:hover {
            background-color: #4f46e5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 NLP Toolkit")
st.caption("Sentiment analysis · Summarization · Named entity recognition")

task = st.selectbox("Choose a task", ["Sentiment Analysis", "Summarization", "Named Entity Recognition"])
text = st.text_area("Enter your text", height=180, placeholder="Paste or type text here...")

@st.cache_resource
def load_pipeline(task_name):
    if task_name == "Sentiment Analysis":
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    if task_name == "Summarization":
        # transformers v5 removed the summarization pipeline; its migration guide recommends a chat model
        return pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    if task_name == "Named Entity Recognition":
        return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

if st.button("Run"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner(f"Running {task.lower()}... (first run downloads the model)"):
            nlp = load_pipeline(task)
            if task == "Sentiment Analysis":
                result = nlp(text)[0]
                st.markdown(
                    f"""<div class="result-card">
                    <b>Label:</b> {result['label']}<br>
                    <b>Confidence:</b> {result['score']:.2%}
                    </div>""",
                    unsafe_allow_html=True,
                )
            elif task == "Summarization":
                messages = [{"role": "user", "content": f"Summarize the following text in two or three sentences:\n\n{text}"}]
                summary = nlp(messages, max_new_tokens=120, do_sample=False)[0]["generated_text"][-1]["content"]
                st.markdown(
                    f"""<div class="result-card">{html.escape(summary)}</div>""",
                    unsafe_allow_html=True,
                )
            elif task == "Named Entity Recognition":
                entities = nlp(text)
                if not entities:
                    st.markdown('<div class="result-card">No entities found.</div>', unsafe_allow_html=True)
                else:
                    rows = "".join(
                        f"<tr><td>{html.escape(e['word'])}</td><td>{e['entity_group']}</td><td>{e['score']:.2%}</td></tr>"
                        for e in entities
                    )
                    st.markdown(
                        f"""<div class="result-card">
                        <table style="width:100%">
                        <tr><th>Entity</th><th>Type</th><th>Confidence</th></tr>
                        {rows}
                        </table>
                        </div>""",
                        unsafe_allow_html=True,
                    )

st.markdown("---")
st.caption("Built with 🤗 Hugging Face Transformers + Streamlit")
