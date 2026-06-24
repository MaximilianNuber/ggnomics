from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

@dataclass
class PcaResult:
    scores: np.ndarray                 # samples × components
    loadings: np.ndarray               # features × components
    explained_variance_ratio: np.ndarray
    feature_names: Optional[np.ndarray] = None
    sample_names: Optional[np.ndarray] = None
    model: Optional[PCA] = None

    def scores_to_pandas(self):
        scores_df = pd.DataFrame(self.scores)
        scores_df.columns = ["pc_"+str(i+1) for i in range(scores_df.shape[1])]
        if self.sample_names is not None:
            scores_df.index = self.sample_names
        return scores_df

    def get_scores(self) -> np.ndarray:
        return self.scores

    def get_loadings(self) -> np.ndarray:
        return self.loadings
    
    def get_explained_variance(self) -> np.ndarray:
        return self.explained_variance_ratio
