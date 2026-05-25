# Deployment Notes

This repository is research-first. Deployment should expose inference and explanation generation without exposing training data.

## Suggested Demo

- Hugging Face Space or Streamlit app
- upload image
- choose model
- return top prediction and attention rollout image

## Safety Notes

- Do not present attention maps as causal proof.
- Do not deploy with private datasets or credentials committed.
- Keep generated outputs in `outputs/` and sample visuals in `assets/`.
