import json
from flask import Flask, request
from flask.json import jsonify
from flask.json.tag import JSONTag
import review_score
import scheduler

app = Flask(__name__)

@app.route('/', methods=['GET'])
def handleModelInput():
    return jsonify({
        "Title":"Jessy Review Model",
        "Description": "This is the model for a machine learning model that can get your input review and output a score for that specific review using machine learning",
        "How to Use" : {
            "Use '/score' route to access the model and pass the review as 'reviewText'",
            "Use /schedule route for the scheduling the trip model and pass the variables as 'time' and 'places'",
            "Use /sentiment route to access the sentiment analysis model and pass the values accordingly"
        },
        "For more info" : "Contact Team Jessy"
    })

@app.route('/score', methods=['POST'])
def handleReviewModelInput():
    reviewText = request.json['reviewText']
    data = str(reviewText)
    score = review_score.getReviewScore(data)
    return score

@app.route('/schedule', methods=['POST'])
def handleSchedulelInput():
    time = request.json['time']
    places = request.json['places']
    timeString = str(time)
    placesString = str(places)
    placesList = scheduler.schedule(timeString, placesString)
    return placesList

@app.route('/sentiment', methods=['POST'])
def handleSentimentInput():
    reviewText = request.json['reviewText']
    data = str(reviewText)
    score = review_score.getReviewScore(data)
    return score

if __name__ == '__main__':
    app.run(debug=True)