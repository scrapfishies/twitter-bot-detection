# Twitter Bot or Not

## Twitter bot detection with supervised machine learning

Whether you're on [Twitter](twitter.com) or stay away from social media altogether, these platforms affect us all - from shaping public discourse to entertainment to spreading information.

The existence of bots on these platforms has gained a lot of attention in recent years, and yet many people are unaware of or misunderstand their presence and purpose on platforms like Twitter.

And so it's important that we start with a [simple working definition of a bot](https://en.wikipedia.org/wiki/Twitter_bot):

> A **Twitter bot is** a **software** bot **that controls a Twitter account via the Twitter API**. It may autonomously perform actions like **tweeting**, **retweeting**, **liking**, **following**, **unfollowing**, or **direct messaging** other users.

Bots are designed for a variety of purposes: they can be creative, helpful, informative, and even funny.

There are, of course, more _nefarious_ bots, which can spread misinformation and scam other users. The presence of these bots can degrade our experience on these platforms and worse: our trust in one another.

I think that **bot _awareness_ is key** to preserving social trust and the integrity of these platforms. And so with that in mind, I wanted to create a tool that could help Twitter users and spectators alike become more bot aware: [Twitter Bot or Not](https://twitter-bot-or-not.herokuapp.com/).

Users can enter in any Twitter handle and see the probability of that account being a bot based on a dozen or so account-level features (more on that to come). These predictions are intended to provide the user with peace of mind with regards to who they're following or interacting with.

---

### Features & target label

The features for this classification model is account-level information of Twitter users - for example, number of followers, tweets, likes, and verification status. I engineered a few additional features to improve the model - for example, a 'tweets-to-followers' metric intended to capture a user's reach.

The target label is account type: bot or human.

I used the [Twitter Bot Accounts](https://www.kaggle.com/davidmartngutirrez/twitter-bots-accounts) dataset from Kaggle, which is comprised of ~37,000 Twitter accounts, labeled as either being bots or humans.

### Model evaluation and results

I chose features and model paramters that would optimize for a balance between precision and recall and a high ROC AUC (area under the curve) score. Essentially I wanted the model to accurately label bots as such, but not by simply labeling _everything_ as a bot.

**Final XGBoost model scores**

- Accuracy: 87.7%
- Precision: 81.3%
- Recall: 80.9%
- ROC AUC: 93.4%

After fitting the model on the full training set, I created a web application with Flask that allowed users to see the bot-likelihood probability of any Twitter account by use of the Twitter API.

Please check it out for yourself! --> [Twitter Bot or Not](https://twitter-bot-or-not.herokuapp.com/)

You can also view my [presentation slides](https://github.com/scrapfishies/twitter-bot-detection/blob/main/presentation_deck.pdf) for the project.

### Suggestions for future work

- Analyze tweet-level data for possible features (good NLP project!)
- **Build a bot to detect bots!**
- Classify _types_ of bots: politcal, fake followers, trolls, etc. (I believe [Botometer](https://botometer.osome.iu.edu/) attempts to do this, too))

---

#### Tools, libraries, & technologies

- Python, Jupyter Notebooks
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- [PostgreSQL](https://www.postgresql.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/#)
- [Twitter API](https://developer.twitter.com/en) and [Tweepy](https://www.tweepy.org/)
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- HTML/CSS, [Bulma](https://bulma.io/)
- [Heroku](https://www.heroku.com/)

#### Sources & references

**Datasets**

- [Kaggle: Twitter Bot Accounts](https://www.kaggle.com/davidmartngutirrez/twitter-bots-accounts)
- [Bot Repository](https://botometer.osome.iu.edu/bot-repository/index.html)

**Heroku Deployment Tutorials**

- [Deploying a flask app to Heroku](https://stackabuse.com/deploying-a-flask-application-to-heroku/)
- [Flask Heroku Example](https://github.com/MirelaI/flask_heroku_example)
