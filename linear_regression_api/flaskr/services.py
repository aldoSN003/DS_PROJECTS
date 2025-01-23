from flask import Blueprint, request, jsonify
import numpy as np
class LinearRegression:
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values
        self.x_mean = np.mean(x_values)
        self.y_mean = np.mean(y_values)

    def slope(self):
        try:


            numerator = sum((x - self.x_mean)*(y - self.y_mean) for x,y in zip(self.x_values,self.y_values))
            denominator = sum((x-self.x_mean)**2 for x in self.x_values)
            b = numerator/denominator
            return round(b,2)

        except Exception as e:
            print(f"Error while computing the slope {str(e)}")
            return False

    def y_intercept(self):
        try:
            a = self.y_mean - (self.slope() * self.x_mean)
            return round(a,2)
        except Exception as e:
            print(f"Error while computingn y intercept {str(e)}")
            return False


    def get_data(self):
        try:
            b = self.slope()
            a = self.y_intercept()
            if b>0:
                equation = f"{a}+{b} x"
            else:
                equation = f"{a}{b} x"
            return jsonify({
                "slope": b,
                "y_intercept": a,
                "equation": equation
            })
        except Exception as e:
            return False

