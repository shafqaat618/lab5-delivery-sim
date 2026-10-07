import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Create Delivery Dataset
# --------------------------------------------------

np.random.seed(42)

# Generate 97 normal delivery times
delivery_times = np.random.normal(
    loc=32,
    scale=7,
    size=97
)

delivery_times = np.round(delivery_times, 1)

# Add 3 intentionally long deliveries
delivery_times = np.concatenate([
    delivery_times,
    [65, 70, 75]
])

# Create Order IDs
order_ids = np.arange(1, 101)

# Create DataFrame
df = pd.DataFrame({
    "OrderID": order_ids,
    "DeliveryTime(min)": delivery_times
})

# Save CSV
df.to_csv("delivery_times.csv", index=False)

print("Delivery dataset created successfully.")

# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

df = pd.read_csv("delivery_times.csv")

# --------------------------------------------------
# 3. Mean and Standard Deviation
# --------------------------------------------------

mean_delivery = df["DeliveryTime(min)"].mean()
std_delivery = df["DeliveryTime(min)"].std()

print("\nReal delivery times:")
print(f"Mean: {mean_delivery:.2f} minutes")
print(f"Standard Deviation: {std_delivery:.2f} minutes")

# --------------------------------------------------
# 4. Simulate 100 Delivery Times
# --------------------------------------------------

simulated_times = np.random.normal(
    loc=mean_delivery,
    scale=std_delivery,
    size=100
)

print(
    f"\nSimulated 100 values from "
    f"Normal(mean={mean_delivery:.2f}, "
    f"std={std_delivery:.2f})"
)

# --------------------------------------------------
# 5. Missing Values
# --------------------------------------------------

missing_values = df.isnull().sum().sum()

print("\nMissing values:")
print(missing_values)

# --------------------------------------------------
# 6. Percentiles
# --------------------------------------------------

q1 = df["DeliveryTime(min)"].quantile(0.25)
median = df["DeliveryTime(min)"].quantile(0.50)
q3 = df["DeliveryTime(min)"].quantile(0.75)

print("\nPercentiles:")
print(f"25th percentile: {q1:.2f}")
print(f"50th percentile: {median:.2f}")
print(f"75th percentile: {q3:.2f}")

# --------------------------------------------------
# 7. IQR Method
# --------------------------------------------------

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("\nIQR:")
print(f"IQR = {iqr:.2f}")

print("\nIQR outlier bounds:")
print(
    f"Lower Bound = {lower_bound:.2f}"
)

print(
    f"Upper Bound = {upper_bound:.2f}"
)

# --------------------------------------------------
# 8. Detect Outliers
# --------------------------------------------------

outliers = df[
    (df["DeliveryTime(min)"] < lower_bound) |
    (df["DeliveryTime(min)"] > upper_bound)
]

print("\nOutliers detected:")

if len(outliers) > 0:
    print(outliers)
else:
    print("No outliers detected.")

# --------------------------------------------------
# 9. Outlier Percentage
# --------------------------------------------------

outlier_percentage = (
    len(outliers) / len(df)
) * 100

print(
    f"\nPercentage flagged as outliers: "
    f"{outlier_percentage:.2f}%"
)

# --------------------------------------------------
# 10. Comparison Histogram
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    df["DeliveryTime(min)"],
    bins=15,
    alpha=0.6,
    label="Real Delivery Times"
)

plt.hist(
    simulated_times,
    bins=15,
    alpha=0.6,
    label="Simulated Delivery Times"
)

plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()

plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")

plt.show()

# --------------------------------------------------
# 11. Boxplot
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.boxplot(
    df["DeliveryTime(min)"]
)

plt.ylabel("Delivery Time (minutes)")
plt.title("Delivery Time Boxplot")

plt.tight_layout()
plt.savefig("delivery_boxplot.png")

plt.show()

print("\nSaved files:")
print("delivery_times.csv")
print("delivery_comparison_histogram.png")
print("delivery_boxplot.png")