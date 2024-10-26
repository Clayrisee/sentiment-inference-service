from joblib import load
from preprocess import Preprocess  # Assuming preprocess is a custom module you've created
from scipy.sparse import csr_matrix
import os
import numpy as np


class InferenceModel(object):
    def __init__(self,
                 model_path: str = os.getenv("MODEL_PATH", "assets/Logistic Regression_v1.0.0.joblib"),
                 vectorizer_path: str = os.getenv("VECTORIZER_PATH", 'assets/vectorizer.pickle'),
                 classes: list = ['negative', 'positive']
                 ) -> None:
        # Load your model
        self.model = load(model_path)
        self.classes = classes
        self.preprocess: Preprocess = Preprocess(vectorizer_path=vectorizer_path)

        # threshold uncertainty
        self.uncertainty_threshold = float(os.getenv("UNCERTAINTY_THRESHOLD", 0.6))
    
    def preprocess(self, text: str) -> csr_matrix:
        return self.preprocess.run_preprocess(text)
    
    def uncertainty_detection(self, probs: np.array):
        """
        Uncertainty Detection using Margin of Confidence
        """
        # First get the most confidence
        probs = probs.tolist()
        probs = sorted(probs, reverse=True)
        most_conf = probs[0]
        next_most_conf = probs[1]
        print(probs)
        uncertainty_score = float(1 - (most_conf - next_most_conf))
        is_uncertain = 1 if uncertainty_score > self.uncertainty_threshold else 0
        return is_uncertain, uncertainty_score

    def postprocess(self, probs: np.array):
       pred_index = probs.argmax()
       prediction = self.classes[pred_index]
       conf = probs[pred_index]
       is_uncertain, uncertainty_score = self.uncertainty_detection(probs)

       return prediction, conf, is_uncertain, uncertainty_score

    def predict(self, text: str) -> dict:
        preprocessed_text = self.preprocess.run_preprocess(text=text)
        probs = self.model.predict_proba(preprocessed_text)[0]
        prediction, conf, is_uncertain, uncertainty_score = self.postprocess(probs=probs)

        return {
            "text": text,
            "prediction": prediction,
            "confidence": float(conf),  # Convert to Python float for JSON serialization
            "is_uncertain": is_uncertain,
            "uncertainty_score": uncertainty_score
        }
