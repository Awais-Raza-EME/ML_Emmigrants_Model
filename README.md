# Emigrant Prediction Based on Profession

This project predicts the number of emigrants from Pakistan based on their professions and the year. The model uses historical data and applies machine learning techniques to provide accurate predictions.

---

## Features
1. **Machine Learning Model**: Utilizes a Random Forest Regressor for predictions.
2. **Data Preprocessing**:
   - One-hot encoding for categorical features (professions).
   - Log transformation of the target variable to handle skewness.
   - Polynomial features for capturing non-linear relationships.
3. **Evaluation Metrics**:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R-squared (R²)

---

## Dataset
The dataset contains information on the number of Pakistani emigrants categorized by:
- **Year** (1981–2023)
- **Profession**

### File Path:
`ML_Emmigrent_Model/number-of-pakistani-emigrants-profession-wise-1981-2023.csv`
adjust if needed.
---

## How It Works
1. **Preprocessing**:
   - Scale the year.
   - Encode professions using one-hot encoding.
   - Apply log transformation to the target variable.
   - Handle outliers using interquartile range (IQR).

2. **Feature Engineering**:
   - Generate polynomial features to capture interactions.
   - Filter out low-importance features based on the trained model's feature importances.

3. **Model Training**:
   - Train a Random Forest Regressor.
   - Hyperparameter tuning (optional) for optimal performance.

4. **Prediction**:
   - Predict emigrant numbers based on user-provided year and profession.

---

## Requirements
Install the required Python libraries:

```bash
pip install numpy pandas scikit-learn
```

---

## Usage
1. **Train the Model**:
   The script preprocesses the data, trains the model, and evaluates performance.

2. **Make Predictions**:
   Use the `predict_emigrants()` function to predict the number of emigrants for a given year and profession:

   ```python
   year_input = 2025
   profession_input = "Doctor"
   predicted_emigrants = predict_emigrants(year_input, profession_input)
   print(f"Predicted number of emigrants: {predicted_emigrants}")
   ```

---

## Results
- **Model Performance**:
  - Training R²: 0.94
  - Testing R²: 0.934
- The model effectively captures trends and produces reliable predictions.

---

## Future Improvements
- Incorporate additional features (e.g., country destination, economic indicators).
- Experiment with advanced models like Gradient Boosting or Neural Networks.
- Automate hyperparameter tuning using GridSearch or RandomizedSearch.

---

## Author
Feel free to modify or extend this project! Let me know if you encounter any issues or have suggestions. 😊
