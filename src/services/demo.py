import gradio as gr
from src.modeling.predict import predict_sentiment
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


print("File Accessed")
# Gradio UI
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a tweet or sentence here..."),
    outputs=[
        gr.Label(num_top_classes=1, label="Predicted Statement"),
        gr.Label(label="Class Probabilities"),
    ],
    title="Sentiment Classifier",
    description="A sentiment classification app using RoBERTa fine-tuned on TweetEval Dataset",
    # token=HF_TOKEN,
)
