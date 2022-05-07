"""
Custom-built tools for testing a fitted model.
"""

def predict_scores(model, X_test, y_test):
    """
    AI is creating summary for predict_scores

    Args:
        model ([type]): [description]
        X_test ([type]): [description]
        y_test ([type]): [description]
    """
    predictions = model.predict(X_test)

    y_ix = y_test.index

    for i in range(0, len(y_test)):
        print(f"Index: {y_ix[i]},",
              f"Predicted quality: {predictions[i]},",
              f"Actual quality: {y_test.iloc[i]}")

