import flask
from twitter_funcs import bot_proba, bot_or_not, get_user_features

app = flask.Flask(__name__)


@app.route('/')
def homepage():
    return flask.render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def make_prediction():
    handle = flask.request.form['handle']
    
    # make predictions with model from twitter_funcs
    user_lookup_message = f'Prediction for @{handle}'

    if get_user_features(handle) == 'User not found':
        prediction = [f'User @{handle} not found', '']

    else: 
        prediction = [bot_or_not(handle), f'Probability of being a bot: {bot_proba(handle)}%']

    return flask.render_template('index.html', prediction=prediction[0], probability=prediction[1], user_lookup_message=user_lookup_message)


# for local dev
if __name__ == '__main__':
    app.run()
