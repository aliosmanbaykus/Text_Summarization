import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import streamlit as st



def summary_model(summary_text):

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
    text = summary_text

    preprocessed_text = text.strip().replace('\n','')
    t5_input_text = 'summarize: ' + preprocessed_text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512, truncation=True).to(device)

    summary_ids = model.generate(tokenized_text, min_length=50, max_length=120)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def main():

   
    summary_text= st.text_area(label='Enter Text to Summarize')
    summary_result=summary_model(summary_text)
    button = st.button("Summarize Text")

    if button:
        st.write(summary_result)
    

if __name__ == '__main__':
    main()