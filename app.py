from __future__ import annotations

import gradio as gr


def status(method: str) -> str:
    return f"{method} demo scaffold is ready. Add a trained/evaluated artifact path before public deployment."


demo = gr.Interface(
    fn=status,
    inputs=gr.Radio(["attention_rollout", "gradcam"], value="attention_rollout", label="Method"),
    outputs="text",
    title="ViT Robustness XAI",
)


if __name__ == "__main__":
    demo.launch()
