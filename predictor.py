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
    # Read the csv for the region that contains the zip code entered and store it in a pandas df
    data_pd = pd.read_csv('data/{}/house-data.csv'.format(s_region), header=0)
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

# Use train_test_split from sklearn to divide the data into a training and testing set
# train_size equals 0.2, meaning the training set will be 80% of the data and the testing set will be 20% of the data
x_tn, x_ts, y_tn, y_ts = train_test_split(
    x, y, test_size=0.2)

# Create a StandardScaler object
scaler = StandardScaler()

# Use the fit_transform method to transform and fit the X training data
x_tn = scaler.fit_transform(x_tn)

# Use the transform method to transform the X testing data
x_ts = scaler.transform(x_ts)

# Create an ANN
# Use keras to create a ANN
home_m_ANN = keras.Sequential()

# Use the add method to add a dense layer. Layer will have 64 neurons and use the relu activation function 
# This is the input layer
home_m_ANN.add(layers.Dense(64, activation='relu', input_shape=(x_tn.shape[1],)))

# Use the add method to add another dense layer. Layer will have 64 neurons and use the relu activation function
# This is a hidden layer
home_m_ANN.add(layers.Dense(64, activation='relu'))

# Use the add method to add another dense layer. Layer will have 64 neurons and use the relu activation function
# This is a hidden layer
home_m_ANN.add(layers.Dense(64, activation='relu'))

# Use the add method to add another dense layer. Layer will have 64 neurons and use the relu activation function
# This is a hidden layer
home_m_ANN.add(layers.Dense(64, activation='relu'))

# Use the add method to add a last dense layer. Layer will have 1 neuron 
# This is the output layer
home_m_ANN.add(layers.Dense(1))

# Use the compile method to prepare the model
# Model will use adam for the optimizer and mean squared error for its loss function
home_m_ANN.compile(optimizer='adam', loss='mean_squared_error')

# Use the fit method to train model
# Model will use the X and y training data.
# Model will have 10000 epochs and a batch size of 6
his = home_m_ANN.fit(x_tn, y_tn, epochs=100, batch_size=6)

### Get the home seller's data
s_beds, s_baths, s_sqft = get_seller_data()

# Create a numpy array from the seller's data
s_d = np.array([[s_beds, s_baths, s_sqft]])

# Use the transform method to transform the seller's data
s_d = scaler.transform(s_d)  # Scale the input

# Use the predict method to get prediction from model using the seller's data
s_p_p = home_m_ANN.predict(s_d)

# Use the predict method to get prediction from model using the testing data
y_pred = home_m_ANN.predict(x_ts)

# Get the acutal prices and the predicted prices for the testing data
a_prices = y_ts.values.flatten()  
p_prices = y_pred.flatten()      

# Calculate percentage off. Actual price minus predicted prices divided by the actual prices, then multiplies by 100
percent_off = abs((a_prices - p_prices) / a_prices) * 100

# Create a pandas df containing the actual prices, predicted prices, and the percents difference from the testing data
t_r_df = pd.DataFrame({
    "Actual Price": a_prices,
    "Predicted Price": p_prices,
    "Percent Off": percent_off
})

# Sort the pandas df by percent off. Then take only the top 5 elements
t_r_df = t_r_df.loc[t_r_df["Percent Off"].abs().sort_values().index].head(5)

# Reset the index for the final DataFrame
t_r_df = t_r_df.reset_index(drop=True)

# Create a pandas df with number of beds, number of baths, square footage, and price for the regional data and seller's data
data = {
    'Metric': ['Beds', 'Baths', 'Square Footage', 'Price'],
    'Averages': [avg_beds, avg_baths, avg_sqft, avg_price],
    'Your Home': [s_beds, s_baths, s_sqft, s_p_p[0][0]]
}
s_r_df = pd.DataFrame(data)

# Set up the subplot grid
f, a = plt.subplots(3, 2, figsize=(14, 10))
f.subplots_adjust(hspace=0.4)
gs = f.add_gridspec(3, 2)

# Create a string to display predicted price
p_price_display = "Your Home Price: ${:.2f}".format(s_p_p[0][0])

# Add a title to the plot
f.suptitle('AI Price Predictor', fontsize=16)

# Add the text containing the predicted price
f.text(0.5, 0.85, p_price_display, ha='center', va='center', fontsize=30, color='green')

# Make sure all cells are off
a[0, 0].axis('off')
a[0, 1].axis('off')
a[1, 0].axis('off')
a[1, 1].axis('off')
a[2, 0].axis('off')
a[2, 1].axis('off')

# Set a title for the table in row 2 (1) column 1 (0)
a[1, 0].set_title("Region Avg. vs. Your Home", fontsize=14)

# Add the table to row 2 (1) column 1 (0)
metrics_table = a[1, 0].table(cellText=s_r_df.values, 
                                colLabels=s_r_df.columns, 
                                cellLoc='center', 
                                loc='center')
metrics_table.auto_set_font_size(False)
metrics_table.set_fontsize(10)
metrics_table.scale(1.2, 1.2)

# Set a title for the table in row 2 (1) column 2 (1)
a[1, 1].set_title("Model Accuracy", fontsize=14)

# Add the table to row 2 (1) column 2 (1)
t_r_table = a[1, 1].table(cellText=t_r_df.round(2).values, 
                                colLabels=t_r_df.columns, 
                                cellLoc='center', 
                                loc='center')
t_r_table.auto_set_font_size(False)
t_r_table.set_fontsize(10)
t_r_table.scale(1.2, 1.2)

# Make a subplot for loss graph
loss_g = f.add_subplot(gs[2, :])

# Plot the loss graph
loss_g.plot(his.history['loss'], label='Training Loss', color='blue', linewidth=2)

# Add title, labels, grid, and a legend to the loss graph
loss_g.set_title('Model Loss Over Epochs', fontsize=18)
loss_g.set_xlabel('Epochs', fontsize=16)
loss_g.set_ylabel('Mean Squared Error (MSE)', fontsize=16)
loss_g.set_ylim(0, max(his.history['loss']) * 1.1)
loss_g.grid(True)
loss_g.legend(loc='upper right', fontsize=14)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
