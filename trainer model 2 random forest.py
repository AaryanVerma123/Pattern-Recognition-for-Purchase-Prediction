import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("enter your dataset here in csv")

# Convert numeric column
df["Discount_Offered in(%)"] = df["Discount_Offered in(%)"].astype(int)

# Create encoders
le_gender = LabelEncoder()
le_income = LabelEncoder()
le_product = LabelEncoder()
le_ad = LabelEncoder()
le_purchase = LabelEncoder()

# Encode categorical columns
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Income_Level"] = le_income.fit_transform(df["Income_Level"])
df["Product_Category"] = le_product.fit_transform(df["Product_Category"])
df["Ad_Interaction"] = le_ad.fit_transform(df["Ad_Interaction"])
df["Purchase"] = le_purchase.fit_transform(df["Purchase"])

# Features & Target
X = df[["Age","Gender","Income_Level","Product_Category",
        "Ad_Interaction","Discount_Offered in(%)","TPP in Past"]]
y = df["Purchase"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=None,         # let trees grow fully
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump((model, le_gender, le_income, le_product, le_ad, le_purchase), f)


print("\nRandom Forest model trained and saved successfully!")
