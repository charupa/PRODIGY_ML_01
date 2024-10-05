import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Load the data
train_path = 'C:/Users/charu/machine_learning_intership/task1/train.csv'  # Replace with actual file path
test_path = 'C:/Users/charu/machine_learning_intership/task1/test.csv'    # Replace with actual file path

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Step 2: Data Preprocessing
# Select relevant features for the model
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
target = 'SalePrice'

# Prepare the training and test data
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
test_predictions = model.predict(X_test)

# Step 5: Generate submission file
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

# Save the submission file
submission.to_csv('house_price_predictions.csv', index=False)

print("Submission file generated: 'house_price_predictions.csv'")
