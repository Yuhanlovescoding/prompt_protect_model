import sys
import pickle
from pathlib import Path
from skops.hub_utils import download

def load_model():
    """
    Loads the prompt injection detection model. If the model is not
    already downloaded locally, it is fetched from the HuggingFace Hub.

    Returns: sklearn.Pipeline: A trained scikit-learn pipeline for detection.
    """
    model_path = 'thevgergroup/prompt_protect'
    if not Path(model_path).is_dir():
        print("Downloading model...")
        download(dst=model_path, repo_id=model_path)
    with open(f"{model_path}/skops-3fs68p31.pkl", "rb") as f:
        return pickle.load(f)

def detect(prompt, model):
    """
    Predicts whether the given prompt is a prompt injection.

    Args:
    prompt (str): The input prompt to check.
    model (sklearn.Pipeline): The trained detection model.

    Returns: bool: True if prompt injection is detected, False otherwise.
    """
    result = model.predict([prompt])[0]
    return result == 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_prompt.py \"your prompt here\"")
        sys.exit(1)

    prompt = sys.argv[1]
    model = load_model()
    is_injection = detect(prompt, model)

    print(f"\n> {prompt}")
    if is_injection:
        print("⚠️  Prompt injection detected")
    else:
        print("✅  No injection detected")

