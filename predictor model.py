import pickle
import pandas as pd

# Load saved model
with open("enter your model here in.pkl", "rb") as f:
    model, le_gender, le_income, le_product, le_ad, le_purchase = pickle.load(f)

# Take user input
age = int(input("Enter Age: "))

gender = input("Enter Gender (Male/Female): ").strip().capitalize()
income = input("Enter Income Level (High/Medium/Low): ").strip().capitalize()

# PRODUCT INPUT 
product_input = input("Enter Product Category: ").strip().lower()

product_map = {
    "designer items": "Designer items",
    "groceries": "Groceries",
    "car": "Car",
    "electronics": "Electronics",
    "beauty products": "Beauty Products"
}

if product_input not in product_map:
    print("\nInvalid product category!")
    print("Choose from: Designer items, Groceries, Car, Electronics, Beauty Products")
    exit()

product = product_map[product_input]

if age < 18 and product == "Car":
    print("\nCustomer is under 18 and cannot legally buy a car.")
    print("Final Decision: No")
    exit()

ad = input("Ad Interaction (Yes/No): ").strip().capitalize()
    
discount = int(input("Discount Offered (%): "))
tpp = int(input("TPP in Past: "))

# Convert to DataFrame
new_customer = pd.DataFrame({
    "Age": [age],
    "Gender": le_gender.transform([gender]),
    "Income_Level": le_income.transform([income]),
    "Product_Category": le_product.transform([product]),
    "Ad_Interaction": le_ad.transform([ad]),
    "Discount_Offered in(%)": [discount],
    "TPP in Past": [tpp]
})

# Prediction
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

result = le_purchase.inverse_transform(prediction)

print("\nWill customer purchase?", result[0])

print("Prediction Probability:", probability)
