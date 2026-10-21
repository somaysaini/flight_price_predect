import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler

class FlightData:
    def __init__(self, file_path="clean_flights.csv"):
        # 1. Data load
        df = pd.read_csv(r'C:\Users\somay\OneDrive\Desktop\solar-power_predection\clean_notebook\clean_flight.csv')
        df = df.drop(columns=["Unnamed: 0", "flight"], errors="ignore")
        
        # 2. x aur y alag
        x = df.drop(columns=["price"])
        y = df["price"]
        
        # 3. Train/test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # 4. Encoding + scaling
        ct = ColumnTransformer([
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), 
             ["airline", "source_city", "destination_city", "departure_time", "arrival_time"]),
            ("stops", OrdinalEncoder(categories=[["zero", "one", "two_or_more"]]), ["stops"]),
            ("class", OrdinalEncoder(categories=[["Economy", "Business"]]), ["class"]),
            ("scale", MinMaxScaler(), ["duration", "days_left"]),
        ])
        
        x_train = ct.fit_transform(x_train)
        x_test = ct.transform(x_test)
        
        # 5. b0 (Intercept) add karna aur class variables (self) mein save karna
        self.x_train = np.insert(x_train, 0, 1, axis=1) 
        self.x_test = np.insert(x_test, 0, 1, axis=1)
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()