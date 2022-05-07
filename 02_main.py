"""
Main scripting environment to express the data preparation process and pipeline structure. Exports
model upon successful execution.
"""

# Local utility imports
import utils.data_handling_lib as dhl
from utils.final_model_constructor import make_final_model, make_comparison_model
from utils.model_testing_lib import predict_scores

raw_dataset = dhl.load_data()
X_train, y_train, X_test, y_test = dhl.strat_split_dataset(X=raw_dataset, label_id="quality")

comparison_slice = X_test[100:125]
comparison_slice_labels = y_test[100:125]

final_model = make_final_model(X=X_train, y=y_train)
final_model_predictions = final_model.predict(X=comparison_slice)
final_model_score = final_model.score(X=X_test, y=y_test)

comparison_model = make_comparison_model(X=X_train, y=y_train)
comparison_model_predictions = comparison_model.predict(X=comparison_slice)
comparison_model_score = comparison_model.score(X=X_test, y=y_test)

predict_scores(final_model, comparison_slice, comparison_slice_labels)

def print_stuff():
    print(f"\nFinal model predictions: {final_model_predictions}",
        f"\nComparison model predictions: {comparison_model_predictions}",
        f"\nActual values: {comparison_slice_labels}",
        f"\nScores\nFinal model: {final_model_score}",
        f"\nComparison model: {comparison_model_score}"
        )
