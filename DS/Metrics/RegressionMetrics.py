import sklearn.metrics as sm


y_true = [3, 0.1, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mae = sm.mean_absolute_error(y_true, y_pred); print(mae)
mse = sm.mean_squared_error(y_true, y_pred); print(mse)
msle = sm.mean_squared_log_error(y_true, y_pred) ; print(msle) 
meae = sm.median_absolute_error(y_true, y_pred); print(meae)
r2 = sm.r2_score(y_true, y_pred); print(r2)
ev1 = sm.explained_variance_score(y_true, y_pred); print(ev1)


y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
ev2 = sm.explained_variance_score(y_true, y_pred, multioutput='raw_values'); print(ev2)
