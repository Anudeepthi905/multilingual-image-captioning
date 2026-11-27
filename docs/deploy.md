# Deploy the demo on Hugging Face Spaces

The demo runs free on a CPU Space using the Docker SDK ([Dockerfile](../Dockerfile)). On CPU the app peaks at about 530 MB of memory, measured locally. That is too close to Streamlit Community Cloud's throttling point, which is why it runs on a Space instead.

## Steps

1. **Create a Hugging Face account** and a write token (Settings → Access Tokens).
2. **Train and publish the model.** Run [the Colab notebook](../notebooks/train_colab.ipynb) with `HF_REPO` set in its last cell. That cell exports the run (weights only, about 26 MB) and uploads it to a model repo. From a machine with a trained run, the same step is:
   ```bash
   python -m src.export --run inceptionv3_v5000_n6000_e20 --push <user>/multilingual-image-captioning
   ```
3. **Deploy the Space:**
   ```bash
   pip install huggingface_hub
   hf auth login
   python deploy/deploy_space.py --space <user>/multilingual-image-captioning \
       --model-repo <user>/multilingual-image-captioning
   ```
   The script uploads only the app files and uses [deploy/space_README.md](../deploy/space_README.md) as the Space's README. It also sets the `MIC_HF_MODEL_REPO` variable, and the Space then builds and starts.
4. **Link it.** Replace `LIVE_URL_TODO` in the README with `https://huggingface.co/spaces/<user>/multilingual-image-captioning`.

## Updating

- **Code:** rerun step 3.
- **Model:** rerun step 2, then restart the Space.

## Notes

- No secrets are needed for a public model repo. For a private one, add `HF_TOKEN` as a Space secret.
- XSRF protection is off in the Dockerfile, because it blocks uploads inside the Spaces iframe. The app keeps no accounts or private state for it to protect.
- The InceptionV3 ImageNet weights (84 MB) are baked into the image, so a cold start doesn't download them.
