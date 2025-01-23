
from flask import Blueprint, request, jsonify
import numpy as np

from flaskr.services import LinearRegression

lr_bp = Blueprint("lr",__name__)
@lr_bp.route("/least_squares_equation", methods=["GET"])
def send_values():
    try:
        data = request.get_json()
        x = np.array(data.get('x'))
        y = np.array(data.get('y'))

        if len(x)!=len(y):
            return jsonify({"error":"Los valores de X y Y deben coincidir"}),400

        lr_model = LinearRegression(x_values=x, y_values=y)

        if lr_model.get_data():
            return lr_model.get_data()



    except Exception as e:
        return jsonify({'message': f'Error al enviar datos: {str(e)}'}), 500
