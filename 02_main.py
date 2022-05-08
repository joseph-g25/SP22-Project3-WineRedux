"""
Main scripting environment to express the data preparation process and pipeline structure. Exports
model upon successful execution.
"""

# Local utility imports
import utils.data_handling_lib as dhl
from final_model_constructor import make_final_model, make_comparison_model
from utils.model_testing_lib import get_r2_score, predict_outputs, get_mse_score

raw_dataset = dhl.load_data()
X_train, y_train, X_test, y_test = dhl.strat_split_dataset(X=raw_dataset, label_id="quality")

comparison_slice = X_test[100:110]
comparison_slice_labels = y_test[100:110]

final_model = make_final_model(X=X_train, y=y_train)
comparison_model = make_comparison_model(X=X_train, y=y_train)

models = {"Final model" : final_model,
          "Comparison model" : comparison_model}

for model_ver, model in models.items():
    print(f"\n{model_ver}\n")
    predict_outputs(model=model,
                    X_test=comparison_slice,
                    y_test=comparison_slice_labels,
                    print_score=True,
                    return_score=False)

    get_mse_score(model=model,
                  X_test=X_test,
                  y_test=y_test,
                  print_score=True,
                  return_score=False)

    get_r2_score(model=model,
                  X_test=X_test,
                  y_test=y_test,
                  print_score=True,
                  return_score=False)

    dhl.save_model(model, model_ver)
