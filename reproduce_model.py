import os
from argparse import ArgumentParser
from tempfile import mkdtemp
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datasets import load_dataset
import skops.hub_utils as hub_utils
from skops.io import dump


class PromptProtectTrainer:
    """
    Trainer class to handle data loading, model training, evaluation,
    saving, and uploading to HuggingFace Hub.
    """
    def __init__(self, config):
        """
        Initializes the trainer with the provided config.
        """
        self.config = config
        self.model = None
        self.dataset = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        """
        Load the dataset from HuggingFace datasets.

        Returns: DatasetDict: HuggingFace dataset dictionary.
        """

        print(f"Loading dataset: {self.config.data}")
        self.dataset = load_dataset(self.config.data)
        return self.dataset

    def prepare_data(self):
        """
        Prepare train/test splits
        """
        train_data = self.dataset['train'].to_pandas()
        test_data = self.dataset['test'].to_pandas()

        self.X_train = train_data['text']
        self.y_train = train_data['label']
        self.X_test = test_data['text']
        self.y_test = test_data['label']

        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")

    def build_model(self):
        """
        Constructs the machine learning pipeline model.

        Returns: scikit-learn pipeline using TF-IDF and LogisticRegression.
        """
        self.model = Pipeline([
            ("vectorize", TfidfVectorizer(max_features=5000)),  # 完全一致
            ("lgr", LogisticRegression())  # 完全一致
        ])
        return self.model

    def train(self):
        """
        Trains the pipeline model using training data.

        Returns: The trained pipeline.
        """
        print("Training model...")
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        """
        Evaluates the trained model on test data.

        Returns: str: Text classification report.
        """
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print("\nModel Evaluation:")
        print(report)
        return report

    def save_model(self):
        """
        Saves the trained model to disk as a .skops file.

        Returns: str: Path to saved model file.
        """
        os.makedirs(self.config.save_dir, exist_ok=True)
        model_path = os.path.join(self.config.save_dir, f"{self.config.model_name}.skops")
        dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        return model_path



def parse_args():
    """
    Parse command line arguments
    """
    parser = ArgumentParser(description="Prompt Injection Detection Model Trainer")
    parser.add_argument("--data", default="deepset/prompt-injections",
                        help="HuggingFace dataset name")
    parser.add_argument("--save-dir", default="models/prompt_protect",
                        help="Directory to save trained model")
    parser.add_argument("--model-name", default="prompt_protect_model",
                        help="Base name for model files")
    parser.add_argument("--repo-id", default="your-username/prompt_protect",
                        help="HF Hub repository ID")
    parser.add_argument("--upload", action="store_true",
                        help="Upload model to HF Hub")
    parser.add_argument("--commit-message", default="Initial model commit",
                        help="Commit message for HF Hub")
    return parser.parse_args()


def main():
    config = parse_args()
    trainer = PromptProtectTrainer(config)

    trainer.load_data()
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
    trainer.evaluate()

    model_path = trainer.save_model()



if __name__ == "__main__":
    main()