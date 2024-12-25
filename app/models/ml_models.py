from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Определение моделей, которые будем использовать
models = {
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "decision_tree": DecisionTreeClassifier
}

def create_model(model_name):
    """Создаёт модель по имени"""
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Unsupported model: {model_name}")
    return model()
