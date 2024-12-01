import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score    
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

#IF YOU WANT TO USE HYPERPARAMETERS AND ALSO PLOT OUTLIERS AND DATA ADD THESE ALSO
# import seaborn as sns   
# import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV

file_path = 'ML_Emmigrent_Prediction_Based_On_Profession/number-of-pakistani-emigrants-profession-wise-1981-2023.csv'
dataFile = pd.read_csv(file_path)

encoder = OneHotEncoder(

)
profession_encoded = encoder.fit_transform(dataFile[['Professsion']]).toarray()
profession_encoded_df = pd.DataFrame(profession_encoded, columns=encoder.get_feature_names_out(['Professsion']))

scaler = StandardScaler()
scaled_year = scaler.fit_transform(dataFile[['Year']])

X = pd.concat([pd.DataFrame(scaled_year, columns=['Year']), profession_encoded_df], axis=1)
y = dataFile['Number of Emigrants']
X['Year'] = X['Year'] * 2  # Double the weight of 'Year'

Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 0.7 * IQR
upper_bound = Q3 + 0.7 * IQR

X_filtered = X[(y >= lower_bound) & (y <= upper_bound)]
y_filtered = y[(y >= lower_bound) & (y <= upper_bound)]

# Apply log transformation to target variable 'y' to handle skewness
y_filtered_log = np.log(y_filtered + 1)  # Adding 1 to avoid log(0)
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered_log, test_size=0.1, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)  # Transform training data
X_test_poly = poly.transform(X_test)       # Transform test data



#  THIS PART WAS USED TO FIND THE BEST PARAMETERS FOR MY RANDOM-FOREST MODEL

#  Parameter grid for Random Forest
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2', None]  # Removed 'auto' as it is not valid for RandomForestRegressor
# }
# GridSearchCV for hyperparameter tuning
# grid_search = GridSearchCV(
#     estimator=RandomForestRegressor(random_state=42),
#     param_grid=param_grid,
#     cv=3,            # 3-fold cross-validation
#     n_jobs=-1,       # Use all available CPU cores
#     verbose=2        # Output detailed messages during the search
# )
# # Fit the grid search on the training data
# grid_search.fit(X_train, y_train)

# # Print the best parameters found
# print("Best hyperparameters:", grid_search.best_params_)
# grid_search.fit(X_train, y_train)
# y_pred = grid_search.predict(X_test)
# best_model = grid_search.best_estimator_


# Train Random Forest model
model = RandomForestRegressor(max_depth=None, max_features=None,n_jobs=-1, min_samples_leaf=2, min_samples_split=2, n_estimators=400, random_state=42)
# model.fit(X_train, y_train)
model.fit(X_train_poly, y_train)
y_pred_log = model.predict(X_test_poly)
y_pred_original = np.exp(y_pred_log) - 1
y_test_original = np.exp(y_test) - 1  # Reverse log transform for comparison



mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)
relative_error = mae / y.mean() * 100  # Percentage error relative to the average value
train_score = model.score(X_train_poly, y_train)
test_score = model.score(X_test_poly, y_test)

y_train_original = np.exp(y_train) - 1              # Back-transform predictions and target to original space
y_test_original = np.exp(y_test) - 1
y_pred_train_original = np.exp(model.predict(X_train_poly)) - 1
y_pred_test_original = np.exp(model.predict(X_test_poly)) - 1
r2_train_original = r2_score(y_train_original, y_pred_train_original)
r2_test_original = r2_score(y_test_original, y_pred_test_original)


# print(f"Original data points: {len(X)}")         THIS IS USED TO CHECK HOW MANY OUTLIERS ARE REMOVED FROM DATA
# print(f"Filtered data points: {len(X_filtered)}")
print(f"Relative MAE: {relative_error}%")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Accuracy on training set (original scale): {:.3f}".format(r2_train_original))
print("Accuracy on test set (original scale): {:.3f}".format(r2_test_original))




# THIS PART IS FOR USER TO USE THE MODEL AFTER THE TRAINING OF MODEL
def predict_emigrants(year_input, profession_input):
    user_data = pd.DataFrame({'Year': [year_input], 'Professsion': [profession_input]})
    profession_encoded_user = encoder.transform(user_data[['Professsion']]).toarray()
    profession_encoded_user_df = pd.DataFrame(profession_encoded_user, columns=encoder.get_feature_names_out(['Professsion']))
    scaled_year_user = scaler.transform(user_data[['Year']])  # Updated scaler logic
    X_user = pd.concat([pd.DataFrame(scaled_year_user, columns=['Year']), profession_encoded_user_df], axis=1)
    X_user_poly = poly.transform(X_user)
    y_pred_log_user = model.predict(X_user_poly)
    y_pred_original_user = np.exp(y_pred_log_user) - 1
    return y_pred_original_user[0]





year_input = int(input("Enter the year: "))
profession_input = input("Enter the profession: ")
predicted_emigrants = predict_emigrants(year_input, profession_input)
print(f"Predicted number of emigrants: {predicted_emigrants}")











