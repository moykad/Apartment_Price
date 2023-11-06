# Import libraries here
import warnings
from glob import glob 
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from IPython.display import VimeoVideo
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted
from category_encoders import OneHotEncoder

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

# Build your `wrangle` function
def wrangle(filepath):
    #1 Subset the data in the CSV file and return only apartments in Mexico City ("Distrito Federal") that cost less than $100,000
    df = pd.read_csv(filepath)
    mask_ba = df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 100000
    df = df[mask_ba & mask_apt & mask_price]
    #2 Remove outliers by trimming the bottom and top 10% of properties in terms of "surface_covered_in_m2"
    low = df["surface_covered_in_m2"].quantile(0.1) 
    high = df["surface_covered_in_m2"].quantile(0.9)
    mask_area=df["surface_covered_in_m2"].between(low,high) 
    df = df[mask_area]
    #3 Create separate "lat" and "lon" columns.
    df[["lat","lon"]]=df["lat-lon"].str.split(",", expand=True).astype(float) 
    
    
    #4 Mexico City is divided into 15 boroughs. Create a "borough" feature from the "place_with_parent_names" column.
    df["borough"]=df["place_with_parent_names"].str.split("|", expand=True)[1]
    
    #5 Drop columns that are more than 50% null values
    df.drop(columns=["surface_total_in_m2", "price_usd_per_m2", "floor", "rooms", "expenses"], inplace=True)
   
    #6 Drop columns containing low- or high-cardinality categorical values
    df.drop(columns=["operation", "property_type", "place_with_parent_names","currency","properati_url"], inplace = True)
    
    #7 Drop any columns that would constitute leakage for the target "price_aprox_usd".
    df.drop(columns = [
    "price",
    "price_aprox_local_currency",
    "price_per_m2"], inplace = True)
    
    #Drop any columns that would create issues of multicollinearity.
    df.drop(columns = ["lat-lon"], inplace = True) 
    return df

# Use this cell to test your wrangle function and explore the data
df = wrangle("data/mexico-city-real-estate-1.csv")
print("df shape:", df.shape)
df.head()

import glob
files = glob.glob('data/mexico-city-real-estate-[0-5].csv')
files

frame = []
for i in files:
    df = wrangle(i)
    frame.append(df)
df = pd.concat(frame)
print(df.info())
df.head()

# Build histogram
plt.hist(df["price_aprox_usd"])


# Label axes
plt.xlabel("Price [$]")
plt.ylabel("Count")

# Add title
plt.title("Distribution of Apartment Prices")

# Don't delete the code below 👇
plt.savefig("images/2-5-4.png", dpi=150)


# Build scatter plot
plt.scatter(x= df["surface_covered_in_m2"], y= df["price_aprox_usd"])


# Label axes
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")

# Add title
plt.title("Mexico City: Price vs. Area");

# Don't delete the code below 👇
plt.savefig("images/2-5-5.png", dpi=150)

# Plot Mapbox location and price
fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="price_aprox_usd",
    mapbox_style="carto-positron",
    labels={"price_aprox_usd": "price_aprox_usd"},
    title="location of the apartments in Mexico City")
fig.show()

# Split data into feature matrix `X_train` and target vector `y_train`.
features = ["surface_covered_in_m2", "lat", "lon", "borough"]
target = "price_aprox_usd"
X_train = df[features]
y_train = df[target]

y_mean = y_train.mean()
y_pred_baseline = [y_mean]*len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)

# Build Model
model = make_pipeline(OneHotEncoder(use_cat_names = True), SimpleImputer(), Ridge())
# Fit model
model.fit(X_train, y_train)

X_test = pd.read_csv("./data/mexico-city-test-features.csv")
print(X_test.info())
X_test.head()

y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()

coefficients = model.named_steps["ridge"].coef_ 
features = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index= features)
feat_imp

# Build bar chart
feat_imp = feat_imp.sort_values(key=abs).tail(15).plot(kind="barh")


# Label axes
plt.xlabel("Importance[USD]")
plt.ylabel("Feature")
# Add title
plt.title("Feature Importance for Apartment Price");
# Don't delete the code below 👇
plt.savefig("images/2-5-13.png", dpi=150)

