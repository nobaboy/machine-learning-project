from sklearn.linear_model import LinearRegression

__all__ = ("linear_regressor",)


# noinspection PyPep8Naming
def linear_regressor(X_train, y_train):
    return LinearRegression().fit(X_train, y_train)
