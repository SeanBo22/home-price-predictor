# Course: CSC510
# Sean Bohuslavsky
# Program to give a user (predict) a selling price for their single family home
# This program only supports a couple zip codes in the Colorado Springs Area

# Use Case
'''
A homeowner is looking to sell their home and does not know how much their home is worth
There are some many people and tools that can help give their home a price, but they do not know who to trust
The want to know what a fair selling price would be for their home using the home data and prices around them
'''

# Import os
import os

# Set Tensorflow loglevel to 3. This will make the program not display warnings to screen
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#Import needed libraries for program 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Initialize Variables
sp_zp_codes = {"South": ["80909", "80910", "80915", "80916"], 
               "Central": ["80917", "80918", "80922", "80923"],
               "West": ["80904", "80907", "80919"], 
               "North": ["80920", "80921", "80924"],
               "East": ["80927", "80938", "80939", "80951"]}

valid_zp_codes = [zip_code for zip_codes in sp_zp_codes.values() for zip_code in zip_codes ]

# Function to get a zip code
# Parameters - None
def get_s_zp_code():

    # Get the zip code
    print("\nThe following are the supported zip codes! I will add more in the future!")
    print("Zip Codes")
    for region, zp_code in sp_zp_codes.items():
        print("{}: {}".format(region, ",".join(zp_code)))
    zp_code = input("\nEnter a valid zip code (Enter 0 to quit): ").strip()
    
    # If user enters 0
    if zp_code == '0':
        exit()
    # If user does not enter a valid zip code
    elif zp_code not in valid_zp_codes:
        # Call get_s_zp_code function again
        zp_code = get_s_zp_code()
    
    for region, zp_codes in sp_zp_codes.items():
        if zp_code in zp_codes:
            s_region = region
    
    return zp_code, s_region

# Function to load the proper data to pandas df
# Parameters - Seller zip code (s_zp - str)
def l_z_data(s_region):
    print("\n**** Loading Housing Data for Zip Code: {} ****".format(s_zp))
    # Read the csv for the zip code entered and store it in a pandas df
    data_pd = pd.read_csv('zillow_data/{}-COS/house-data.csv'.format(s_region), header=0)
    return data_pd

# Function to get the number of beds for seller's home
# Parameters - None
def get_seller_beds():

    # Set a threshold. The max amount of number of beds in zip code plus 1
    max = h_df['Beds'].max() + 1

    # Get a valid number of beds
    try:
        s_n_beds = int(input("\nEnter the number of beds for your home: (Enter 0 to quit): ").strip())
    except:
        print("Please enter a valid number")
        s_n_beds = get_seller_beds()
    if s_n_beds == 0:
        exit()
    elif s_n_beds < 0 or s_n_beds > max:
        print("Please enter a value greater than 0 and no greater than {}".format(max))
        s_n_beds = get_seller_beds()
    
    return s_n_beds

# Function to get the number of baths for seller's home
# Parameters - None
def get_seller_baths():

    # Set a threshold. The max amount of number of baths in zip code plus 1
    max = h_df['Baths'].max() + 1

    # Get a valid number of baths
    try:
        s_n_baths = int(input("\nEnter the number of baths for your home: (Enter 0 to quit): ").strip())
    except:
        print("Please enter a valid number")
        s_n_baths = get_seller_baths()
    if s_n_baths == 0:
        exit()
    elif s_n_baths < 0 or s_n_baths > max:
        print("Please enter a value greater than 0 and no greater than {}".format(max))
        s_n_baths = get_seller_baths()
    
    return s_n_baths

# Function to get the square feet for seller's home
# Parameters - None
def get_seller_sqft():

    # Set a threshold. The max square feet in zip code plus 500
    max = h_df['Footage'].max() + 500

    # Get a valid number of baths
    try:
        s_sqft = int(input("\nEnter the square footage for your home: (Enter 0 to quit): ").strip())
    except:
        print("Please enter a valid number")
        s_sqft = get_seller_sqft()
    if s_sqft == 0:
        exit()
    elif s_sqft < 0 or s_sqft > max:
        print("Please enter a value greater than 0 and no greater than {}".format(max))
        s_sqft = get_seller_sqft()
    
    return s_sqft

# Function to get all data points for seller's home
# Parameters - None
def get_seller_data():
    print("Please enter in the following attributes for your home!")

    # Get number of beds
    s_beds = get_seller_beds()

    # Get number of baths
    s_baths = get_seller_baths()

    # Get square feet
    s_sqft = get_seller_sqft()

    return s_beds, s_baths, s_sqft


print("**** Home Price Predictor for Colorado Springs ****")

# Get a zip code
s_zp, s_region = get_s_zp_code()

# Load the proper data to a pandas df
h_df = l_z_data(s_region)

# Create x. A data frame containing only the data points (Beds, Baths, and Footage)
x = h_df[['Beds', 'Baths', 'Footage']]

#Create y. A data frame with the target (price)
y = h_df[['Price']]

# Calculate the average price, beds, baths, and sqft for the homes in the zip code
avg_price = round(h_df['Price'].mean(),2)
avg_beds = round(h_df['Beds'].mean(),1)
avg_baths = round(h_df['Baths'].mean(), 1)
avg_sqft = round(h_df['Footage'].mean(),2)
print(avg_price)
print(avg_beds)
print(avg_baths)
print(avg_sqft)

### End of Cell 1

### Beginning of Cell 2
indices = np.arange(len(x))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    x, y, indices, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Create the ANN model
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the model
model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1)

### End of Cell 2

### Beginning of Cell 3
s_beds, s_baths, s_sqft = get_seller_data()


# Create an input array for the model prediction
input_data = np.array([[s_beds, s_baths, s_sqft]])
input_data_scaled = scaler.transform(input_data)  # Scale the input

# Make a prediction
predicted_price = model.predict(input_data_scaled)
# Display the predicted price
print("\nThe predicted price for your home is: ${:,.2f}".format(predicted_price[0][0]))

### End of Cell 3

### Beginning of Cell 4
y_pred = model.predict(X_test)
actual_prices = y_test.values.flatten()  
predicted_prices = y_pred.flatten()      

# Calculate percentage off
percent_off = abs((actual_prices - predicted_prices) / actual_prices) * 100

# Create the results table as a DataFrame
results_df = pd.DataFrame({
    "Actual Price": actual_prices,
    "Predicted Price": predicted_prices,
    "Percent Off": percent_off
})
# Sort the DataFrame by the absolute values of "Percent Off" and select the top 5 rows
results_df = results_df.loc[results_df["Percent Off"].abs().sort_values().index].head(5)

# Reset the index for the final DataFrame
results_df = results_df.reset_index(drop=True)

# Example data for home metrics
data = {
    'Metric': ['Beds', 'Baths', 'Square Footage'],
    'Averages': [avg_beds, avg_baths, avg_sqft],
    'Your Home': [s_beds, s_baths, s_sqft]  # Example values for your home
}
metrics_df = pd.DataFrame(data)

# Set up the subplot grid
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display the results DataFrame as a table on the left subplot
axs[0].axis('off')
metrics_table = axs[0].table(cellText=metrics_df.values, 
                             colLabels=metrics_df.columns, 
                             cellLoc='center', 
                             loc='center')
metrics_table.auto_set_font_size(False)
metrics_table.set_fontsize(10)
metrics_table.scale(1.2, 1.2)
axs[0].set_title("Home Metrics Comparison", fontsize=14)

# Display the metrics DataFrame as a table on the right subplot
axs[1].axis('off')
results_table = axs[1].table(cellText=results_df.round(2).values, 
                             colLabels=results_df.columns, 
                             cellLoc='center', 
                             loc='center')
results_table.auto_set_font_size(False)
results_table.set_fontsize(10)
results_table.scale(1.2, 1.2)
axs[1].set_title("Prediction Results Comparison", fontsize=14)

# Adjust layout to add space between the tables
plt.subplots_adjust(wspace=0.4)  # Adjust horizontal space between tables

# Show the combined plot
plt.tight_layout()
plt.show()








