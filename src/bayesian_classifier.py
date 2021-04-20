# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import xgboost as xgb

from ax import optimize
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator


class BayesianxgClassifier(BaseEstimator):
    def __init__(self, CV: int = 5, trial_number: int = 40):

        super().__init__()

        self.trial_number = trial_number

        self.__model = None
        self.__best_params = None
        self.__CV = CV

    def fit(self, X, y):

        try:
            n_classes = int(y.max() + 1)
            xgb_train = xgb.DMatrix(X, label=y)

            def evaluate(
                eta: float,
                gamma: float,
                max_depth: int,
                subsample: float,
                l1: float,
                l2: float,
                n_estimators: int,
            ):
                """
                Evaluates CV function helper
                """
                params = {
                    "eta": eta,
                    "gamma": gamma,
                    "max_depth": max_depth,
                    "subsample": subsample,
                    "lambda": l1,
                    "alpha": l2,
                    "objective": "multi:softprob",
                    "num_class": n_classes,
                }

                result = xgb.cv(
                    params,
                    xgb_train,
                    num_boost_round=n_estimators,
                    nfold=self.__CV,
                    metrics=["mlogloss"],
                )

                return result.loc[n_estimators - 1, "test-mlogloss-mean"]

            opt_param = [
                {
                    "name": "eta",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.01, 1],
                },
                {
                    "name": "gamma",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0, 100],
                },
                {
                    "name": "max_depth",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [1, 9],
                },
                {
                    "name": "subsample",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.5, 1],
                },
                {
                    "name": "lambda",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0, 100],
                },
                {
                    "name": "alpha",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0, 100],
                },
                {
                    "name": "n_estimators",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [10, 200],
                },
            ]

            opt_eval = lambda p: evaluate(
                p["eta"],
                p["gamma"],
                p["max_depth"],
                p["subsample"],
                p["lambda"],
                p["alpha"],
                p["n_estimators"],
            )
            best, value, _, _ = optimize(
                parameters=opt_param,
                evaluation_function=opt_eval,
                minimize=True,
                total_trials=self.trial_number,
            )

            self.__best_params = {
                "eta": best["eta"],
                "gamma": best["gamma"],
                "max_depth": best["max_depth"],
                "subsample": best["subsample"],
                "lambda": best["lambda"],
                "alpha": best["alpha"],
                "objective": "multi:softprob",
                "num_class": n_classes,
            }

            self.__model = xgb.train(
                self.__best_params, xgb_train, num_boost_round=best["n_estimators"]
            )

            self.__best_params["n_estimators"] = best["n_estimators"]
            self.fitted = True

        except Exception as e:
            raise RuntimeError("Fit error")

        return self

    @property
    def feature_importances_(self):
        if self.fitted:
            return list(self.__model.get_score(importance_type="weight").values())

        raise RuntimeError("Model not fitted")

    def predict(self, X, y=None):

        if self.fitted:
            return self.__model.predict(xgb.DMatrix(X))

        raise RuntimeError("Model not fitted")

    def predict_proba(self, X, y=None):

        if self.fitted:
            return self.__model.predict_proba(xgb.DMatrix(X))

        raise RuntimeError("Model not fitted")

    def score(self, X, y):

        if self.fitted:
            return -log_loss(y, self.__model.predict(xgb.DMatrix(X)))

        raise RuntimeError("Model not fitted")
