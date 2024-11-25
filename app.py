from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\kalpe\Desktop\Air\model4.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from the form
        age = int(request.form['Age'])
        customer_type = request.form['CustomerType']
        travel_type = request.form['TypeOfTravel']
        travel_class = request.form['Class']
        flight_distance = float(request.form['FlightDistance'])
        departure_delay = float(request.form['DepartureDelay'])
        arrival_delay = float(request.form['ArrivalDelay'])
        onboard_service = int(request.form['OnboardService'])
        seat_comfort = int(request.form['SeatComfort'])
        leg_room_service = int(request.form['LegRoomService'])
        cleanliness = int(request.form['Cleanliness'])
        food_and_drink = int(request.form['FoodAndDrink'])
        inflight_service = int(request.form['InflightService'])
        wifi_service = int(request.form['InflightWifiService'])
        entertainment = int(request.form['InflightEntertainment'])
        baggage_handling = int(request.form['BaggageHandling'])

        # Map categorical values to numeric
        customer_type_mapping = {'Loyal Customer': 1, 'Disloyal Customer': 0}
        travel_type_mapping = {'Business Travel': 1, 'Personal Travel': 0}
        class_mapping = {'Business': 2, 'Economy': 1, 'Economy Plus': 0}

        customer_type = customer_type_mapping.get(customer_type, -1)
        travel_type = travel_type_mapping.get(travel_type, -1)
        travel_class = class_mapping.get(travel_class, -1)

        # Construct the input DataFrame
        input_data = pd.DataFrame([{
            'Age': age,
            'Customer Type': customer_type,
            'Type of Travel': travel_type,
            'Class': travel_class,
            'Flight Distance': flight_distance,
            'Departure Delay': departure_delay,
            'Arrival Delay': arrival_delay,
            'On-board Service': onboard_service,
            'Seat Comfort': seat_comfort,
            'Leg Room Service': leg_room_service,
            'Cleanliness': cleanliness,
            'Food and Drink': food_and_drink,
            'In-flight Service': inflight_service,
            'In-flight Wifi Service': wifi_service,
            'In-flight Entertainment': entertainment,
            'Baggage Handling': baggage_handling
        }])

        # Make the prediction
        prediction = model.predict(input_data)

        # Return the predicted value
        return render_template('index.html', prediction_text=f'Predicted Satisfaction: {prediction[0]}')

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
