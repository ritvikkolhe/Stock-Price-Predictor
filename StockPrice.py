#Program to predict the price of the google stocks for the next 15 days using the Quandl dataset.
#Made by: Ritvik Ajay Kolhe

import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("\nProcessing... Please wait\n")

df = quandl.get("WIKI/GOOGL")           #Google stock dataset from Quandl

df = df.iloc[-500: ,-5: ]               #Slicing the required data

df['Prediction'] = df[['Adj. Close']].shift(int(-15))       #A new column with data shifted 15 units up for prediction

X = df.drop(columns=['Prediction'])             #Seperating the data on which price is going to be predicted

#Dividing the dataset into test and train
y_train=np.array(df['Prediction'][:-15])
X_train=np.array(X[:-15])
X_test=np.array(X[-15:])

#Training the model
model = LinearRegression()
model.fit(X_train,y_train)

forecast_prediction = model.predict(X_test)           #Prediction made

#To create a dataframe of predicted price
forecast_prediction=[df['Adj. Close'][-1]]+forecast_prediction.tolist()
Pred_data={"Forecast":forecast_prediction}
dates = pd.date_range('20180327', periods=16)
Pred_df = pd.DataFrame(data=Pred_data, index=dates)

print("Done.\n")
print(Pred_df[-15:])          #Displaying predicted price

#For ploting the graph of Date vs Price
df['Adj. Close'].plot()
Pred_df['Forecast'].plot()
plt.legend(loc=4)
plt.title("Google Stock Price")
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()          #To display graph to the user
