# imports
import json
import sys
import numpy
import sqlite3
import os

import pandas as pd
from tqdm import tqdm
from time import sleep
from typing import List
from sqlite3 import Error

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from config import path_to_data, db_name, directory_with_files

data = []

with open(path_to_data, 'r') as file: # opens a file and puts each json into a list
    for line in file:
        data.append(json.loads(line.strip()))


def retrieving_full_text(dictionary: dict) -> str:
    '''
    Function retrieves entire text if tweet is truncated, otherwise it returns unchanged text.
    :param dictionary: dictionary that we are working with
    :param f_text: initializes the returned text
    :returns: extended text or leaves it unchanged
    '''
    try:
        f_text = ''
        if dictionary['truncated'] == True:
            f_text = dictionary['extended_tweet'].get('full_text')
        else:
            f_text = dictionary['text']
        return f_text
    except:
        print('Wrong structure of a given tweet.')


def retrieving_user_mentions(dictionary: dict) -> List[int]:
    """
    Function retrieves user mentions from the tweet.
    :param dictionary: dictionary that we are working with
    :param user_ment: list with user mentions
    :returns: list with ints that represent id of mentioned users, if no users were mentioned then it returns an empty list
    """
    user_ment = []
    for user in dictionary['entities']['user_mentions']:
        try:
            user_ment.append(user['id'])
        except: 
            break
    return user_ment


def retrieving_hashtags(dictionary: dict) -> List[str]:
    """
    Function retrieves used hashtangs in a tweet.
    :param dictionary: dictionary that we are working with
    :param hash_lst: list with used hashtags
    :returns: list with strings that represent each hashtag, if no hashtags were used then it returns an empty list
    """
    hash_lst = []
    for hashtag in dictionary['entities']['hashtags']:
        try:
            hash_lst.append(hashtag['text'])
        except:
            break
    return hash_lst


def retrieving_symbols(dictionary: dict) -> List[str]:
    """
    Funtion retrieves used symbols in a tweet.
    :param dictionary: dictionary that we are working with
    :param symb_lst: list with used symbols
    :returns: list with strings that represent each symbol, if no symbols were used then it returns an empty list
    """
    symb_lst = []
    for symbol in dictionary['entities']['symbols']:
        try:
            symb_lst.append(symbol['text'])
        except:
            break
    return symb_lst
            

retrieving_user_mentions(data[1])  # testing retrieving_user_mentions() function
retrieving_hashtags(data[5]) # testing retrieving_hashtags() function
retrieving_full_text(data[7]) # testing retrieving_full_text() function
retrieving_full_text(data[6009]) # tesing retrieving_full_text() function
retrieving_symbols(data[239]) # testing retrieving_symbols() function

orig_tweet_lst = [] # List of tuples of all users in this file

for tweet in data:
    try: # we use try and except in thin for loop, because some of the tweets are deleted and have no stored info
        if tweet['in_reply_to_status_id'] is None and tweet['is_quote_status'] == False:
        
            tweet_id = tweet['id']
            user_id = tweet['user']['id']
            full_text = retrieving_full_text(tweet)
            timestamp_ms = tweet['timestamp_ms']
            user_mentions = retrieving_user_mentions(tweet)
            hashtags = retrieving_hashtags(tweet)
            symbols = retrieving_symbols(tweet)
            quote_count = tweet['quote_count']
            reply_count = tweet['reply_count']
            favorite_count = tweet['favorite_count']
            retweet_count = tweet['retweet_count']

            connected = (tweet_id, user_id,full_text,timestamp_ms, user_mentions, hashtags, symbols, quote_count, reply_count, favorite_count, retweet_count)

            orig_tweet_lst.append(connected)
    except:
        continue
        
len(orig_tweet_lst) # number of tweets that are original


def go_through_files(directory_with_files, db_name):
    conn = create_connection(db_name)
    cursor = conn.cursor()
    for n, filename in enumerate(os.listdir(directory_with_files)):
        f = os.path.join(directory_with_files, filename)
        if os.path.isfile(f):
            write_json_to_db(f, cursor)
    conn.commit()
    conn.close()
        
def write_json_to_db(file, cursor):
    data=[]
    with open(file, 'r', encoding='utf-8') as f:
        for n, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except ValueError:
                print(n, line)
            
    parse_data(data, cursor)

def insert(where, data, cursor):
    values_placeholder = ", ".join(["?"] * len(data))
    into_placeholder = tuple(data.keys())
    variables = tuple(data.values())
    query = f"""
    INSERT OR IGNORE INTO {where} {into_placeholder}
    VALUES ({values_placeholder})
    """
    cursor.execute(query, variables)


def tweet_classification(data):
    keys=data.keys()
    check_for_none=True
    if 'in_reply_to_status_id' in keys:
        if data['in_reply_to_status_id']!= None:
            return reply_processing(data)
    if 'retweeted_status' in keys:
        return retweet_processing(data)
    if 'quoted_status' in keys:
        return quote_processing(data)
    if 'delete' in keys:
        return delete_processing(data)
    return original_processing(data)

def original_processing(data):
    base_data = base_retrieving(data)
    text_data = text_retrieving(data)
    geo_data = geo_retrieving(data)
    tweet_data, entities = base_processing(text_data, base_data)
    tweet_data['tweet_type'] = 'original'
    return tweet_data, entities, geo_data, None

def quote_processing(data):
    special ={}
    base_data = base_retrieving(data)
    text_data = text_retrieving(data)
    geo_data = geo_retrieving(data)
    tweet_data, entities = base_processing(text_data, base_data)
    tweet_data['tweet_type'] = 'quote'
    special['quote_id'] = base_data['tweet_id']    
    special['quote_of_status_id'] = data['quoted_status_id']
    return tweet_data, entities, geo_data, special

def retweet_processing(data):
    special ={}    
    base_data = base_retrieving(data)
    text_data = text_retrieving(data['retweeted_status'])
    geo_data = geo_retrieving(data)
    tweet_data, entities = base_processing(text_data, base_data)
    tweet_data['tweet_type'] = 'retweet'
    special['retweet_id'] = base_data['tweet_id']
    special['retweet_of_status_id'] = data['retweeted_status']['id']
    return tweet_data, entities, geo_data, special

def reply_processing(data):
    special ={}   
    base_data = base_retrieving(data)
    text_data = text_retrieving(data)
    geo_data = geo_retrieving(data)
    tweet_data, entities = base_processing(text_data, base_data)
    tweet_data['tweet_type'] = 'reply'
    special['reply_id'] = base_data['tweet_id']
    special['reply_to_user_id'] = data['in_reply_to_user_id']
    special['reply_to_status_id'] = data['in_reply_to_status_id']
    return tweet_data, entities, geo_data, special


def base_processing(text_data, base_data):
    tweet_data ={}
    entities = {}
    tweet_data['tweet_id'] = base_data['tweet_id']
    tweet_data['user_id'] = base_data['user_id']
    tweet_data['timestamp_ms'] = base_data['timestamp_ms']
    tweet_data['text'] = text_data['text']
    tweet_data['lang'] = base_data['lang']
    entities['hashtags'] = text_data['hashtags']
    entities['user_mentions'] = text_data['user_mentions']
    entities['symbols'] = text_data['symbols']
    return tweet_data, entities
    
def delete_processing(data):
    return 'delete', 'delete', 'delete', 'delete'

def text_retrieving(data):
    keys = data.keys()
    text_data ={}
    if 'extended_tweet' in keys:
        text_data['text'] = data['extended_tweet']['full_text']
        
        raw_hashtags = data['extended_tweet']['entities']['hashtags']
        hashtags=[]
        for hashtag in raw_hashtags:
            hashtags.append(hashtag['text'])
        text_data['hashtags'] = hashtags
        
        raw_user_mentions = data['extended_tweet']['entities']['user_mentions']
        user_mentions = []
        for user_mention in raw_user_mentions:
            user_mentions.append(user_mention['id'])
        text_data['user_mentions'] = user_mentions
        
        raw_symbols = data['extended_tweet']['entities']['symbols']
        symbols = []
        for symbol in raw_symbols:
            symbols.append(symbol['text'])
        text_data['symbols'] = symbols
    else:
        text_data['text'] = data['text']
        
        raw_hashtags = data['entities']['hashtags']
        hashtags=[]
        for hashtag in raw_hashtags:
            hashtags.append(hashtag['text'])
        text_data['hashtags'] = hashtags
        
        raw_user_mentions = data['entities']['user_mentions']
        user_mentions = []
        for user_mention in raw_user_mentions:
            user_mentions.append(user_mention['id'])
        text_data['user_mentions'] = user_mentions
        
        raw_symbols = data['entities']['symbols']
        symbols = []
        for symbol in raw_symbols:
            symbols.append(symbol['text'])
        text_data['symbols'] = symbols
    text_data['user_id'] = data['user']['id']
    return text_data

def base_retrieving(data):
    keys = data.keys()
    return_data={}
    
    return_data['timestamp_ms'] = data['timestamp_ms']
    return_data['tweet_id'] = data['id']
    return_data['user_id'] = data['user']['id']
    return_data['lang'] = data['lang']
    return return_data
    
    
def geo_retrieving(data):
    return_data={}
    if data['place'] != None:
        return_data['tweet_geo_id'] = data['id']
        return_data['full_name'] = data['place']['full_name']
        return_data['country'] = data['place']['country']
        return_data['country_code'] = data['place']['country_code']
        return return_data

    else:
        return None
    
def user_retrieving(data):
    keys = data.keys()
    user_data = {}
    if 'user' in keys:
        user_data['user_id'] = data['user']['id']
        user_data['name'] = data['user']['name']
        user_data['screen_name'] = data['user']['screen_name']
        user_data['location'] = data['user']['location']
        user_data['followers_count'] = data['user']['followers_count']
        user_data['friends_count'] = data['user']['friends_count']
    return user_data

def parse_data(data, cursor):
    for n,line in enumerate(data):
        tweet_data, entities, geo_data, special = tweet_classification(line)
        
        if tweet_data == 'delete':
            continue
        insert('tweets', tweet_data, cursor)
        if geo_data != None:
            geo_data['tweet_type'] = tweet_data['tweet_type']
            insert('tweets_geo', geo_data, cursor)
            
        if tweet_data['tweet_type'] == 'retweet':
            insert('retweets', special, cursor)
        if tweet_data['tweet_type'] == 'reply':
            insert('replies', special, cursor)
        if tweet_data['tweet_type'] == 'quote':
            insert('quotes', special, cursor)
        insert('users', user_retrieving(line), cursor)
        
        for a_hashtag in entities['hashtags']:
            hashtag={}
            hashtag['tweet_id'] = tweet_data['tweet_id']
            hashtag['text'] = a_hashtag
            insert('hashtags', hashtag, cursor)
        for a_symbol in entities['symbols']:
            symbol={}
            symbol['tweet_id'] = tweet_data['tweet_id']
            symbol['text'] = a_symbol
            insert('symbols', symbol, cursor)
        for a_user_mention in entities['user_mentions']:
            user_mention={}
            user_mention['tweet_id'] = tweet_data['tweet_id']
            user_mention['text'] = a_user_mention
            insert('user_mentions', user_mention, cursor)

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

def create_all_tables(db_file):
    create_tweets_table = """
    CREATE TABLE IF NOT EXISTS tweets(
        tweet_id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        timestamp_ms INTEGER NOT NULL, 
        text TEXT NOT NULL,
        lang TEXT NOT NULL,
        tweet_type TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """

    create_replies_table = """
    CREATE TABLE IF NOT EXISTS replies(
        reply_id INTEGER PRIMARY KEY,
        reply_to_status_id INTEGER NOT NULL,
        reply_to_user_id INTEGER NOT NULL,
        FOREIGN KEY (reply_id) REFERENCES tweets(tweet_id)
    );
    """

    create_retweets_table = """
    CREATE TABLE IF NOT EXISTS retweets(
        retweet_id INTEGER PRIMARY KEY,
        retweet_of_status_id INTEGER NOT NULL,
        FOREIGN KEY (retweet_id) REFERENCES tweets(tweet_id)
    );
    """

    create_quotes_table = """
    CREATE TABLE IF NOT EXISTS quotes(
        quote_id INTEGER PRIMARY KEY,
        quote_of_status_id INTEGER NOT NULL,
        FOREIGN KEY (quote_id) REFERENCES tweets(tweet_id)
    );
    """

    create_users_table = """
    CREATE TABLE IF NOT EXISTS users(
        user_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        screen_name TEXT NOT NULL,
        location TEXT,
        followers_count INTEGER NOT NULL,
        friends_count INTEGER NOT NULL
    );
    """

    create_hashtags_table = """
    CREATE TABLE IF NOT EXISTS hashtags(
        hashtag_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tweet_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY (tweet_id) REFERENCES tweets(tweet_id)
    );
    """
    create_user_mentions_table = """
    CREATE TABLE IF NOT EXISTS user_mentions(
        user_mention_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tweet_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY (tweet_id) REFERENCES tweets(tweet_id)
    );
    """
    create_symbols_table = """
    CREATE TABLE IF NOT EXISTS symbols(
        symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tweet_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY (tweet_id) REFERENCES tweets(tweet_id)
    );
    """

    create_tweets_geo_table = """
    CREATE TABLE IF NOT EXISTS tweets_geo(
        tweet_geo_id INTEGER NOT NULL PRIMARY KEY,
        full_name TEXT NOT NULL, 
        country TEXT NOT NULL,
        country_code TEXT NOT NULL,
        tweet_type TEXT NOT NULL,
        FOREIGN KEY (tweet_geo_id) REFERENCES tweets(tweet_id)
    );
    """    
    conn = create_connection(db_file)
    cursor = conn.cursor()
    cursor.execute(create_users_table)
    cursor.execute(create_tweets_table)
    cursor.execute(create_tweets_geo_table)
    cursor.execute(create_replies_table)
    cursor.execute(create_retweets_table)
    cursor.execute(create_quotes_table)
    cursor.execute(create_hashtags_table)
    cursor.execute(create_symbols_table)
    cursor.execute(create_user_mentions_table)
    conn.close()


create_connection(db_name)
create_all_tables(db_name)
go_through_files(directory_with_files, db_name) # go through files 





