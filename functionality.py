import sys
from skops.io import load
from pathlib import Path
from skops.io import get_untrusted_types

def load_model():
    """
    Loads locally trained model.

    Returns: our trained detection model.
    """
    model_path = 'models/prompt_protect/prompt_protect_model.skops'
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}.\n"
            "Please train your model first by running reproduce_model.py"
        )

    untrusted = get_untrusted_types(file=model_path)
    return load(model_path, trusted=list(untrusted))


def detect(prompt, model):
    """
    Predicts whether the given prompt is a prompt injection.

    Args:
    prompt (str): The input prompt to check.
    model: The trained detection model.

    Returns: bool: True if prompt injection is detected, False otherwise.
    """
    return model.predict([prompt])[0] == 1

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

