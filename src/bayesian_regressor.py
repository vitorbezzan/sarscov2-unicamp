# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

from ax import optimize

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score


class BayesianxgRegressor(BaseEstimator):

    def __init__(self, CV: int = 5, trial_number: int = 40):

        super().__init__()

        self.trial_number = trial_number
        self.__CV = CV

        self.__model = None
        self.__best_params = None

    def fit(self, X, y):

        try:

            def evaluate(
                eta: float,
                max_depth: int,
                subsample: float,
                n_estimators: int,
            ):
                """
                Evaluates CV function helper
                """
                model = XGBRegressor(learning_rate=eta, n_estimators=n_estimators, subsample=subsample,
                                                  max_depth=max_depth)
                score = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=self.__CV, n_jobs=-1)
                return score.mean()

            opt_param = [
                {
                    "name": "eta",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.1, 1],
                },
                {
                    "name": "max_depth",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [3, 6],
                },
                {
                    "name": "subsample",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.7, 1],
                },
                {
                    "name": "n_estimators",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [5, 300],
                },
            ]

            opt_eval = lambda p: evaluate(
                p["eta"],
                p["max_depth"],
                p["subsample"],
                p["n_estimators"]
            )
            best, value, _, _ = optimize(
                parameters=opt_param,
                evaluation_function=opt_eval,
                minimize=False,
                total_trials=self.trial_number,
            )

            self.__best_params = {
                "eta": best["eta"],
                "max_depth": best["max_depth"],
                "subsample": best["subsample"],
                "n_estimators": best["n_estimators"]
            }

            self.__model = XGBRegressor(learning_rate=self.__best_params["eta"],
                                                     n_estimators=self.__best_params["n_estimators"],
                                                     subsample=self.__best_params["subsample"],
                                                     max_depth=self.__best_params["max_depth"])
            self.__model.fit(X, y)

            self.fitted = True

        except Exception as e:
            raise RuntimeError(str(e))

        return self

    def predict(self, X, y=None):

        if self.fitted:
            return self.__model.predict(X)

        raise RuntimeError("Model not fitted")

    def score(self, X, y):

        if self.fitted:
            return -mean_squared_error(y, self.__model.predict(X))

        raise RuntimeError("Model not fitted")
