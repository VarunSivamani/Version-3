from flask import Flask, render_template, jsonify, request
import threading
import queue

app = Flask(__name__)

# Store training data
current_data = {
    "loss": 0,
    "accuracy": 0,
    "epoch": 0,
    "batch": 0
}
loss_history = []
accuracy_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    try:
        data = request.json
        # Update current data
        current_data.update(data)
        # Store both loss and accuracy
        loss_history.append(data['loss'])
        accuracy_history.append(data['accuracy'])
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error in update: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_data')
def get_data():
    try:
        if current_data["epoch"] > 0:  # Only return if we have data
            return jsonify(current_data)
        return jsonify({"status": "no_data"})
    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_loss_history')
def get_loss_history():
    try:
        return jsonify({
            "loss_history": loss_history,
            "accuracy_history": accuracy_history
        })
    except Exception as e:
        print(f"Error in get_loss_history: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        loss_history.clear()
        accuracy_history.clear()
        current_data.update({
            "loss": 0,
            "accuracy": 0,
            "epoch": 0,
            "batch": 0
        })
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error in clear_history: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 