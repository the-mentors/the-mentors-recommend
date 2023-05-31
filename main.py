from flask import Flask
from flask import request

from filter import *

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# 주소 형식: http://localhost:5000/api/v1/recommends?user_id=4860
@app.route('/api/v1/recommends')
def recommends():
    user_id = request.args.get('user_id', "null")
    l = filter(user_id)
    return { "recommend" : l }

@app.route('/training')
def train():
    user_id = request.args.get('user_id', "null")
    training(user_id)
    return {"complete" : "success"}


if __name__ == "__main__":
    app.run(host='0.0.0.0')
    app.run(debug=True)
