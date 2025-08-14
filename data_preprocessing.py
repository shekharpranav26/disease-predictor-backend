import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.symptom_columns = []
        self.label_encoder = LabelEncoder()
        self.symptoms_data = None
        self.precautions_data = None

    # ---------- helpers ----------
    @staticmethod
    def _norm(sym: str) -> str:
        s = str(sym).lower().strip()
        s = re.sub(r'[^a-z0-9\s_]', ' ', s)
        s = re.sub(r'\s+', '_', s).strip('_')
        return s

    def _normalize_symptom(self, symptom: str) -> str:
        return self._norm(symptom)

    def _find_best_symptom_match(self, user_symptom_norm: str):
        # 1) exact
        if user_symptom_norm in self.symptom_columns:
            return user_symptom_norm

        # 2) partial contains (both ways)
        for col in self.symptom_columns:
            if user_symptom_norm in col or col in user_symptom_norm:
                return col

        # 3) token overlap (underscore → space)
        u_tokens = set(user_symptom_norm.replace('_', ' ').split())
        best = None
        best_overlap = 0
        for col in self.symptom_columns:
            c_tokens = set(col.replace('_', ' ').split())
            overlap = len(u_tokens & c_tokens)
            if overlap > best_overlap:
                best = col
                best_overlap = overlap
        return best if best_overlap > 0 else None

    # ---------- main loaders ----------
    def load_and_preprocess_data(self):
        try:
            self.symptoms_data = pd.read_csv('data/DiseaseAndSymptoms.csv', dtype=str).fillna('')
            self.precautions_data = pd.read_csv('data/Disease precaution.csv', dtype=str).fillna('')

            logger.info(f"Loaded dataset with shape: {self.symptoms_data.shape}")

            # Symptom_* columns
            symptom_cols = [c for c in self.symptoms_data.columns if c.lower().startswith('symptom')]
            if not symptom_cols:
                raise ValueError("No Symptom_* columns found in the dataset")

            # All unique textual symptoms
            unique_symptoms = set()
            for c in symptom_cols:
                unique_symptoms.update(self.symptoms_data[c].astype(str).str.strip().str.lower().tolist())
            unique_symptoms = {s for s in unique_symptoms if s and s != 'nan'}

            # Normalized column names
            norm_map = {orig: self._norm(orig) for orig in unique_symptoms}
            norm_columns = sorted(set(norm_map.values()))

            # One-hot symptom matrix
            X = pd.DataFrame(0, index=self.symptoms_data.index, columns=norm_columns)
            for c in symptom_cols:
                for idx, val in self.symptoms_data[c].astype(str).items():
                    v = val.strip().lower()
                    if v and v != 'nan':
                        n = norm_map.get(v)
                        if n:
                            X.at[idx, n] = 1

            self.symptom_columns = list(X.columns)

            # Target
            y_text = self.symptoms_data['Disease'].astype(str).str.strip()
            # Drop rows with no active symptoms
            valid = X.sum(axis=1) > 0
            X = X.loc[valid].reset_index(drop=True)
            y_text = y_text.loc[valid].reset_index(drop=True)

            y = self.label_encoder.fit_transform(y_text)

            logger.info(f"Preprocessed data -> X: {X.shape}, y: {y.shape}, features: {len(self.symptom_columns)}")
            return X, y, self.symptom_columns

        except Exception as e:
            logger.error(f"Error in load_and_preprocess_data: {str(e)}")
            raise

    # ---------- user input ----------
    def preprocess_user_symptoms(self, user_symptoms):
        """
        Returns:
          - numpy array shape (1, n_features) with 0/1
          - list of matched symptom column names
        """
        try:
            if not self.symptom_columns:
                # ensure columns are available
                self.load_and_preprocess_data()

            n_features = len(self.symptom_columns)
            symptom_vector = np.zeros(n_features, dtype=np.float32)

            # Normalize incoming symptoms
            normalized = [self._normalize_symptom(s) for s in (user_symptoms or []) if str(s).strip()]
            matched = []

            # Build matches and vector
            for s in normalized:
                best = self._find_best_symptom_match(s)
                if best and best in self.symptom_columns:
                    if best not in matched:
                        matched.append(best)
                        idx = self.symptom_columns.index(best)
                        symptom_vector[idx] = 1.0

            logger.info(f"User symptoms: {user_symptoms}")
            logger.info(f"Matched symptoms: {matched}")
            logger.info(f"Active symptoms count: {float(symptom_vector.sum())}")

            return symptom_vector.reshape(1, -1), matched

        except Exception as e:
            logger.error(f"Error preprocessing user symptoms: {str(e)}")
            raise

    # ---------- utilities ----------
    def get_available_symptoms(self):
        try:
            if not self.symptom_columns:
                self.load_and_preprocess_data()
            return sorted(self.symptom_columns)
        except Exception:
            return []

    def get_available_diseases(self):
        try:
            if not getattr(self.label_encoder, 'classes_', None) is not None:
                self.load_and_preprocess_data()
            return sorted(self.label_encoder.classes_.tolist())
        except Exception:
            return []

    def get_precautions(self, disease: str):
        try:
            if self.precautions_data is None:
                self.precautions_data = pd.read_csv('data/Disease precaution.csv', dtype=str).fillna('')

            df = self.precautions_data.copy()
            df.columns = df.columns.str.strip()

            row = df[df['Disease'].str.lower() == str(disease).lower()]
            if row.empty:
                return []

            # columns like Precaution_1 / Precaution 1 / precaution_2 etc.
            cols = [c for c in df.columns if re.match(r'(?i)^precaution[_\s]*\d+$', c)]
            out = []
            for c in sorted(cols, key=lambda x: int(re.findall(r'\d+', x)[0])):
                v = str(row.iloc[0][c]).strip()
                if v and v.lower() != 'nan':
                    out.append(v)
            return out
        except Exception as e:
            logger.error(f"Error getting precautions for {disease}: {str(e)}")
            return []

    def decode_prediction(self, encoded_prediction):
        try:
            return self.label_encoder.inverse_transform([int(encoded_prediction)])[0]
        except Exception as e:
            logger.error(f"Error decoding prediction: {str(e)}")
            return "Unknown"
