import numpy as np


def crowd_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors))) if len(errors) else 0.0
    rmse = float(np.sqrt(np.mean(errors * errors))) if len(errors) else 0.0
    mask = actual != 0
    mape = (
        float(np.mean(np.abs(errors[mask] / actual[mask]))) * 100.0
        if mask.any()
        else 0.0
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
