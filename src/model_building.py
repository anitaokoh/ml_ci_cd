import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

class CaliforniaHousingModel:
    def __init__(self):
        self.model = LinearRegression()
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_pred = [None] * 5

    def load_and_split_data(self):
        california = fetch_california_housing()
        data = pd.DataFrame(california.data, columns=california.feature_names)
        data['PRICE'] = california.target

        X = data.drop('PRICE', axis=1)
        y = data['PRICE']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def save_model(self, filename='trained_model.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    # def predict(self):
    #     self.y_pred = self.model.predict(self.X_test)
    #     return self.y_pred

    # def evaluate(self):
    #     mse = mean_squared_error(self.y_test, self.y_pred)
    #     r2 = r2_score(self.y_test, self.y_pred)
    #     return mse, r2

if __name__ == "__main__":
    model = CaliforniaHousingModel()
    model.load_and_split_data()
    model.train()
    model.save_model()
    # mse, r2 = model.evaluate()
    
    # print(f"Mean Squared Error: {mse}")
    # print(f"R^2 Score: {r2}")
