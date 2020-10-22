
/* Creates an SQL database for Twitter accounts */
/* Then creates a Table for twitter_human_bots_dataset */

CREATE DATABASE twitter_accounts; 

\connect twitter_accounts;

CREATE TABLE human_bots(
    index INT,
    created_at TEXT, 
    default_profile BOOLEAN, 
    default_profile_image BOOLEAN,
    acct_description TEXT,
    favourites_count INT,
    followers_count INT,
    friends_count INT,
    geo_enabled BOOLEAN,
    id BIGINT,
    lang TEXT, 
    acct_location TEXT, 
    profile_background_image_url TEXT, 
    profile_image_url TEXT,
    screen_name TEXT, 
    statuses_count INT, 
    verified BOOLEAN, 
    average_tweets_per_day FLOAT, 
    account_age_days INT, 
    account_type TEXT
);

\copy human_bots FROM 'data_files/twitter_human_bots_dataset.csv' DELIMITER ',' CSV HEADER;

