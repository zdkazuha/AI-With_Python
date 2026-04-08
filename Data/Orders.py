import numpy as np
import pandas as pd


data_csv = pd.read_csv("./assets/Orders.csv")

df_csv = pd.DataFrame(data_csv)

df_csv["TotalAmount"] = df_csv["Quantity"] * df_csv["Price"]

# 1

data = {
    'OrderId': [1001, 1002, 1003],
    'Customer': ["Alice", "Bob", "Alice"],
    'Product': ["Laptop", "Chair", "Mouse"],
    'Category': ["Electronics","Furniture", "Electronics"],
    'Quantity': [1,2,3],
    'Price': [1500, 180, 25],
    'OrderDate': ["2023-06-01", "2023-06-03", "2023-06-05"]
}

df = pd.DataFrame(data)

df["OrderDate"] = pd.to_datetime(df["OrderDate"])

# 2

df["TotalAmount"] = df["Quantity"] * df["Price"]

# 3 a

sum = df["TotalAmount"].sum()

# print(sum)

# 3 b

abs = df["TotalAmount"].mean()

# print(abs)

# 3 c ()

customer_orders = df.groupby("Customer")["OrderId"].count()

# print(customer_orders)

# 4

filtered_df = df[df["Price"] > 500]

# print("Filtered: ", filtered_df)

# 5

filtered_df = df.sort_values(by = "OrderDate", ascending=False)

# print("Filtered: ", filtered_df)

# 6

filtered_df = df.query('"2023-06-05" <= OrderDate <= "2023-06-10"')

# print("Filtered: ", filtered_df)

# 7 a

category_quantity = df.groupby("Category")["Quantity"].count()

# print(category_quantity)

# 7 b

category_totalAmount = df.groupby("Category")["TotalAmount"].sum()

# print(category_totalAmount)

# 8

customer_orders = df_csv.groupby("Customer")["TotalAmount"].sum().reset_index()

top_3 = customer_orders.sort_values(by="TotalAmount", ascending=False).head(3)

print(top_3)