"""
Custom-built tools for testing a fitted model.
"""

from sklearn.metrics import mean_squared_error, r2_score

def predict_outputs(model,
                    X_test,
                    y_test,
                    print_score=False,
                    return_score=True):
    """
    Get and/or print predictions and actuals for a given model on iterable input and output data.

    Args:
        model (sklearn model): sklearn machine learning model with .predict method.
        X_test (list/series/DataFrame/array): Set of input values to use to predict target values.
        y_test (list/series/DataFrame/array): Set of actual target values.
        print_score (bool, optional): Print predictions and actual values to console. Defaults to False.
        return_score (bool, optional): Return prediction list. Defaults to True.

    Returns:
        [list]: List of prediction values
    """
    predictions = model.predict(X_test)

    y_ix = y_test.index

    if print_score is True:
        for i in range(0, len(y_test)):
            print(f"Index: {y_ix[i]},",
                f"Predicted quality: {predictions[i]},",
                f"Actual quality: {y_test.iloc[i]}")
    if return_score is True:
        return predictions

def get_mse_score(model, X_test, y_test, squared=False, print_score=False, return_score=True):
    """
    Get and/or print MSE score for a list of predictions.

    Args:
        model (model): sklearn model with .predict method
        X_test (list/series/DataFrame/array): Test data features
        y_test (list/series/DataFrame/array): Test data labels
        squared (bool, optional): MSE (True) or RMSE (False). Defaults to False.
        print_score (bool, optional): Print score to console. Defaults to False.
        return_score (bool, optional): Return score. Defaults to True.

    Returns:
        [float]: MSE score
    """
    predictions = model.predict(X_test)

    score = mean_squared_error(y_true=y_test, y_pred=predictions, squared=squared)
    if print_score is True:
        print(f"(R)MSE Score: {score}")

    if return_score is True:
        return score

def get_r2_score(model,
                X_test,
                y_test,
                print_score=False,
                return_score=True):
    """
    AI is creating summary for get_r2_score

    Args:
        model (model): sklearn model with .predict method.
        X_test (list/series/DataFrame/array): Test data features.
        y_test (list/series/DataFrame/array): Test data labels.
        print_score (bool, optional): Print score to console. Defaults to False.
        return_score (bool, optional): Return score. Defaults to True.

    Returns:
        [float]: R^2 score.
    """
    predictions = model.predict(X_test)

    score = r2_score(y_true=y_test, y_pred=predictions)
    if print_score is True:
        print(f"R^2 Score: {score}")

    if return_score is True:
        return score


def loop_print(iterable):
    """
    Print each member of an iterable object.

    Args:
        iterable (Any): A 2D iterable such as a list or series.
    """
    for obj in iterable:
        print(obj)
