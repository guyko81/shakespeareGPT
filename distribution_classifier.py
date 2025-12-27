"""
Distribution Classifier (Single LGBM)

A scikit-learn-compatible classifier that predicts full probability distributions
over discrete classes. When sigma=0, it uses one-hot encoding (hard targets).
When sigma>0, it uses soft targets with Gaussian smoothing.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from lightgbm import LGBMRegressor
from scipy.ndimage import gaussian_filter1d


class DistributionClassifier(BaseEstimator, ClassifierMixin):
    """
    Predicts probability distributions over discrete classes using a single 
    gradient boosted model.
    
    Parameters
    ----------
    n_bins : int or 'auto', default='auto'
        Number of grid points (classes).
        - If 'auto': Uses unique values in y as the grid (recommended for classification).
        - If int: Creates a linspace grid with n_bins points.
    
    sigma : float, default=0.0
        The standard deviation of the Gaussian kernel used to generate soft targets.
        - If 0.0: Uses one-hot encoding (hard targets) - true classification.
        - If > 0: Uses Gaussian smoothing around true class (soft targets).
    
    output_smoothing : float, default=0.0
        Standard deviation for Gaussian smoothing of the output distribution.
        Set to 0.0 to disable (default for classification).
    
    n_estimators : int, default=100
        Number of boosting trees.
    
    learning_rate : float, default=0.1
        Boosting learning rate.
        
    random_state : int or None, default=None
        Random seed.
    
    categorical_feature : list of int or 'auto', default=None
        List of feature indices to treat as categorical.
        - If None: No categorical features.
        - If 'auto': Let LightGBM detect categorical features.
        - If list of int: Indices of categorical features in X.
    
    device : str, default='cpu'
        Device to use for training. 'cpu' or 'gpu'.
        
    **kwargs : dict
        Additional parameters passed to LGBMRegressor.
    """
    
    def __init__(
        self,
        n_bins='auto',
        sigma=0.0,
        output_smoothing=0.0,
        n_estimators=100,
        learning_rate=0.1,
        random_state=None,
        categorical_feature=None,
        device='cpu',
        **kwargs
    ):
        self.n_bins = n_bins
        self.sigma = sigma
        self.output_smoothing = output_smoothing
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.categorical_feature = categorical_feature
        self.device = device
        self.lgbm_kwargs = kwargs

    def fit(self, X, y, sample_weight=None):
        """
        Fit the classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class values.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample.
        """
        # 1. Prepare Data
        self._is_dataframe = isinstance(X, pd.DataFrame)
        if self._is_dataframe:
            self.feature_names_in_ = X.columns.tolist()
            X_array = X.values
            y_array = np.asarray(y)
        else:
            X_array = X
            y_array = np.asarray(y)
            
        X_array, y_array = check_X_y(X_array, y_array, accept_sparse=False, dtype=np.float64)
        self.n_features_in_ = X_array.shape[1]
        n_samples = X_array.shape[0]
        
        # Validate sample_weight
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != n_samples:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, expected {n_samples}")

        # 2. Define the Grid (Classes)
        if self.n_bins == 'auto':
            # Use unique values as grid
            self.classes_ = np.sort(np.unique(y_array))
            self.grid_ = self.classes_.astype(np.float64)
            self.n_bins_ = len(self.classes_)
        else:
            y_min = float(np.min(y_array))
            y_max = float(np.max(y_array))
            self.grid_ = np.linspace(y_min, y_max, self.n_bins)
            self.classes_ = self.grid_
            self.n_bins_ = self.n_bins

        # 3. Expand Dataset
        n_total_rows = n_samples * self.n_bins_
        X_final = np.empty((n_total_rows, self.n_features_in_ + 1), dtype=X_array.dtype)
        
        # Fill feature columns using broadcasting
        X_final[:, :-1].reshape(n_samples, self.n_bins_, -1)[:] = X_array[:, None, :]
        
        # Fill grid_point column using broadcasting
        X_final[:, -1].reshape(n_samples, self.n_bins_)[:] = self.grid_
        
        # 4. Generate Targets
        if self.sigma == 0.0:
            # One-hot encoding (hard targets)
            # Find closest grid point for each y value
            y_indices = np.abs(y_array[:, None] - self.grid_[None, :]).argmin(axis=1)
            targets = np.zeros((n_samples, self.n_bins_), dtype=np.float64)
            targets[np.arange(n_samples), y_indices] = 1.0
        else:
            # Soft targets with Gaussian smoothing
            diff_sq = (y_array[:, None] - self.grid_[None, :]) ** 2
            sigma_sq = self.sigma ** 2
            targets = np.exp(-diff_sq / (2 * sigma_sq))
            # Normalize each row
            row_sums = targets.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            targets = targets / row_sums
            
        y_targets = targets.ravel()
        
        # 5. Feature Names
        if self._is_dataframe:
            feature_names = self.feature_names_in_ + ['grid_point']
        else:
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)] + ['grid_point']
            
        # 6. Configure LightGBM
        params = {
            'objective': 'cross_entropy',
            'metric': 'cross_entropy',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'verbose': -1,
            'device': self.device,
        }
        if self.device == 'gpu':
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
        params.update(self.lgbm_kwargs)
        
        self.model_ = LGBMRegressor(**params)
        
        # Expand sample_weight to match expanded dataset
        if sample_weight is not None:
            sample_weight_expanded = np.repeat(sample_weight, self.n_bins_)
        else:
            sample_weight_expanded = None
        
        # Handle categorical features (grid_point column is added at the end, indices unchanged)
        cat_feature_param = self.categorical_feature
            
        self.model_.fit(
            X_final, 
            y_targets, 
            sample_weight=sample_weight_expanded, 
            feature_name=feature_names,
            categorical_feature=cat_feature_param
        )

        return self

    def predict_proba(self, X):
        """
        Returns probability distribution over classes for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples for which to predict probabilities.
        
        Returns
        -------
        probabilities : array of shape (n_samples, n_classes)
            Probability distribution for each sample.
        """
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        X_array = check_array(X_array, accept_sparse=False)
        n_samples = X_array.shape[0]
        
        # 1. Expand X for prediction
        X_expanded = np.repeat(X_array, self.n_bins_, axis=0)
        grid_tile = np.tile(self.grid_, n_samples)
        
        if self._is_dataframe:
            feature_names = self.feature_names_in_
        else:
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]
            
        X_df_expanded = pd.DataFrame(X_expanded, columns=feature_names)
        X_df_expanded['grid_point'] = grid_tile
        
        # 2. Predict
        pred_scores = self.model_.predict(X_df_expanded)
        
        # 3. Reshape to (n_samples, n_bins)
        scores_matrix = pred_scores.reshape(n_samples, self.n_bins_)
        
        # 4. Normalize
        row_sums = scores_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        distributions = scores_matrix / row_sums
        
        # 5. Apply output smoothing if enabled
        if self.output_smoothing > 0:
            distributions = gaussian_filter1d(distributions, sigma=self.output_smoothing, axis=1)
            row_sums = distributions.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            distributions = distributions / row_sums
        
        return distributions

    def predict_distribution(self, X):
        """
        Returns grid points and probability distribution for each sample.
        
        Returns
        -------
        grid : array of shape (n_classes,)
            Class values.
        
        distributions : array of shape (n_samples, n_classes)
            Probability distribution for each sample.
        """
        return self.grid_, self.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels (mode of distribution).
        
        Returns
        -------
        predictions : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        max_indices = np.argmax(proba, axis=1)
        return self.classes_[max_indices]

