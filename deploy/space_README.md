---
title: Multilingual Image Captioning
emoji: 🗣️
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Describe a photo in 9 languages, as text and speech
---

# Multilingual Visual Description

Upload a photo (or try a sample) and get an English caption, a translation into one of
9 languages, and spoken audio. InceptionV3 encoder, Bahdanau-attention GRU decoder,
Google Translate and Google Text-to-Speech.

An academic project; captions can be wrong. Source code, training notebook and
limitations: https://github.com/Anudeepthi905/multilingual-image-captioning

This Space is deployed with `deploy/deploy_space.py` from that repository. The trained
model is downloaded at startup from the Hugging Face model repo named in the Space
variable `MIC_HF_MODEL_REPO`.
