import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


MODEL_PATH = "car-gpt2-lora"  # no leading "./"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


pipe = load_model()

st.title("Car Description Generator")
st.caption("Fine-tuned on 1,000,000 car marketing descriptions")

prompt = st.text_input("Enter a car prompt:", "Introducing the all-new BMW M5")

max_length = st.slider("Max Length", 40, 200, 80)
temperature = st.slider("Creativity", 0.3, 1.5, 0.9)
repetition_penalty = st.slider("Repetition Penalty", 1.0, 3.0, 1.8)

if st.button("Generate"):
    with st.spinner("Generating..."):
        output = pipe(prompt, max_length=max_length, do_sample=True,
                      temperature=temperature, top_p=0.92,
                      repetition_penalty=repetition_penalty)[0]["generated_text"]
    st.write(output)

