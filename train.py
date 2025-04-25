from sklearn.pipeline import Pipeline

from skops import card, hub_utils

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
from skops.io import dump
from tempfile import mkdtemp, mkstemp
import sklearn
from argparse import ArgumentParser


data = "deepset/prompt-injections"
save_directory = "models"
model_name = "prompt_protect_model"
repo_id = "thevgergroup/prompt_protect"
upload = False
commit_message = "Initial commit"

X_train, X_test, y_train, y_test = None, None, None, None


def load_data(data):
    """
    Loads a HuggingFace dataset by name. The dataset must include
    'train' and 'test' splits and contain 'text' and 'label' fields.

    Args:data (str): Name or path of the dataset.

    Returns:DatasetDict: A dictionary containing train/test splits.
    """
    dataset = load_dataset(data)
    return dataset


def split_data(dataset):
    """
    Splits the loaded dataset into training and testing sets. This function assumes
    that the dataset is already split into 'train' and 'test'.

    Populates global variables: X_train, y_train, X_test, y_test
    """
    global X_train, X_test, y_train, y_test
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    X_train = df_train['text']
    y_train = df_train['label']
    X_test = df_test['text']
    y_test = df_test['label']


def train_model(X_train, y_train):
    """
    Trains a logistic regression model using TF-IDF vectorization.

    Args:
    X_train (List[str]): Training texts.
    y_train (List[int]): Training labels.

    Returns: Pipeline: A trained sklearn pipeline.
    """
    model = Pipeline(
        [
            ("vectorize", TfidfVectorizer(max_features=5000)),
            ("lgr", LogisticRegression()),
        ]
    )
    model.fit(X_train, y_train)

    return model


def evaluate_model(model):
    """
    Evaluates the trained model on the test dataset and returns a classification report.

    Args: model (Pipeline): The trained scikit-learn pipeline.

    Returns: str: Classification report as a string.
    """
    global X_train, X_test, y_train, y_test
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="deepset/prompt-injections",
                        help="Dataset to use for training, expects a huggingface dataset with train and test splits and text / label columns")
    parser.add_argument("--save_directory", type=str, default="models/thevgergroup",
                        help="Directory to save the model to")
    parser.add_argument("--model_name", type=str, default="prompt_protect_model",
                        help="Name of the model file, will have .skops extension added to it")
    parser.add_argument("--repo_id", type=str, default="thevgergroup/prompt_protect", help="Repo to push the model to")
    parser.add_argument("--upload", action="store_true",
                        help="Upload the model to the hub, must be a contributor to the repo")
    parser.add_argument("--commit-message", type=str, default="Initial commit",
                        help="Commit message for the model push")

    args = parser.parse_args()

    if any(vars(args).values()):
        data = args.data
        save_directory = args.save_directory
        model_name = args.model_name
        repo_id = args.repo_id
        upload = args.upload
        commit_message = args.commit_message

    dataset = load_data(data)
    split_data(dataset)
    model = train_model(X_train=X_train, y_train=y_train)
    report = evaluate_model(model)
    print(report)

    # save model locally
    model_path = os.path.join(save_directory)
    print("Saving model to", model_path)
    os.makedirs(model_path, exist_ok=True)

    model_file = os.path.join(model_path, f"{model_name}.skops")

    dump(model, file=model_file)

    if upload:
        local_repo = mkdtemp(prefix="skops-")
        print("Creating local repo at", local_repo)
        hub_utils.init(model=model_file,
                       dst=local_repo,
                       requirements=[f"scikit-learn={sklearn.__version__}"],
                       task="text-classification",
                       data=X_test.to_list(),
                       )

        hub_utils.add_files(__file__, dst=local_repo, exist_ok=True)

        hub_utils.push(source=local_repo, repo_id=repo_id, commit_message=commit_message)