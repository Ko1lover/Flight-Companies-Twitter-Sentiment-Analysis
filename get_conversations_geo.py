# imports 
import os
import ast
import csv
import json
import sqlite3
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

from sqlite3 import Error
from statistics import mean
from treelib import Node, Tree
from  matplotlib.colors import LinearSegmentedColormap

# create a colormap which is going to be used later for visualization
cmap=LinearSegmentedColormap.from_list('rg',['#D5F5E3', 'red', 'red'], N=256)

# constant variables and paths 

directory_with_files = '../data'
db_name = './test.db'
trees_file = './replies_trees.csv'
convs_file = './conversations.json'
airlines=[56377143, 106062176, 18332190, 22536055, 124476322, 26223583, 2182373406, 38676903, 1542862735, 253340062, 218730857, 45621423, 20626359]

# This function is used to run a query on a database and return the results as a pandas dataframe
# Parameters:
#     query (str): the query to be run on the database
#     conn (sqlite3.Connection): the connection to the database
# Returns:
#     pandas.DataFrame: the results of the query as a dataframe
def run_query(query: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Runs the given query on the given database connection and returns the results as a pandas dataframe.
    """
    df = pd.read_sql_query(query, conn)
    return df

def create_connection(db_file):
    """
    Create a database connection to a SQLite database.
    
    This function attempts to connect to a SQLite database specified by the db_file parameter.
    If the connection is successful, it prints the SQLite version and returns the connection object.
    If the connection fails, it prints the error message and returns None.
    
    Parameters:
    - db_file (str): The path to the database file to connect to.
    
    Returns:
    - conn (sqlite3.Connection): The connection object to the SQLite database if successful, None otherwise.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

def find_conv(user_id, check_id, db_name, conn):
    """
    Finds and constructs a dictionary representing the conversation chain starting from a specific tweet.
    
    This function queries the database to find all replies to a given tweet (identified by check_id) and recursively
    finds replies to those replies, building a nested dictionary structure that represents the entire conversation
    chain. Each key in the dictionary is a tuple of (user_id, tweet_id), and the value is a list of dictionaries,
    each containing 'user_id' and 'tweet_id' of the replies.
    
    Parameters:
    - user_id (int): The user ID of the original tweet's author.
    - check_id (int): The tweet ID of the original tweet to start the conversation chain from.
    - db_name (str): The name of the database file (unused in this function but part of the signature for consistency).
    - conn (sqlite3.Connection): The connection object to the SQLite database.
    
    Returns:
    - dict: A dictionary representing the conversation chain, where each key is a tuple of (user_id, tweet_id),
      and each value is a list of dictionaries with 'user_id' and 'tweet_id' of the replies.
    
    Note:
    - This function is marked as redundant, indicating there may be a more efficient or updated method for achieving
      the same result.
    """
    print('done')
    query = f"""
        SELECT reply_id, reply_to_status_id, reply_to_user_id, user_id, tweet_id
        FROM replies, tweets
        WHERE reply_id=tweet_id AND reply_to_status_id = {check_id}
        """
    df = run_query(query, conn)
    past_replies_dict={}
    now_reply_ids = list(df['reply_id'])
    past_replies_dict[(user_id, check_id)] = df[['user_id', 'tweet_id']].to_dict('records')
    while len(now_reply_ids) != 0:
        past_replies_dict, now_reply_ids = check_for_replies(past_replies_dict,now_reply_ids, conn)
    return past_replies_dict

def check_for_replies(past_replies_dict, new_replies_list, conn):
    """
    Updates the conversation dictionary with replies to the tweets in the new_replies_list.
    
    This function queries the database for replies to each tweet ID in new_replies_list. It updates
    the past_replies_dict with these replies, mapping each original tweet to its replies. The function
    returns the updated dictionary and a list of new reply IDs to facilitate further recursive searches
    for replies to these new replies.
    
    Parameters:
    - past_replies_dict (dict): The current dictionary of tweet conversations, where each key is a tuple
      of (reply_to_user_id, reply_to_status_id) and each value is a list of dictionaries containing 'user_id'
      and 'tweet_id' of the replies.
    - new_replies_list (list): A list of tweet IDs to find replies for.
    - conn (sqlite3.Connection): The database connection object.
    
    Returns:
    - tuple: A tuple containing the updated past_replies_dict and a list of new reply IDs (now_reply_ids).
    """
    now_reply_ids = []
    for a_id in new_replies_list:
        query = f"""
        SELECT reply_id, reply_to_status_id, reply_to_user_id, user_id, tweet_id
        FROM replies, tweets
        WHERE reply_id=tweet_id AND reply_to_status_id = {a_id}
        """
        return_df = run_query(query, conn)
        returned_ids = list(return_df['reply_id'])
        now_reply_ids += returned_ids
        if len(return_df) != 0:
            past_replies_dict[(return_df['reply_to_user_id'][0], return_df['reply_to_status_id'][0])] = return_df[['user_id', 'tweet_id']].to_dict('records')
        
    return past_replies_dict, now_reply_ids

def get_tweets_with_replies(db_name):
    """
    Retrieves tweets that have received replies from the database.
    
    This function connects to the SQLite database specified by db_name. It executes a SQL query to select
    tweets that are of type 'original' and have at least one reply. The query results are returned as a
    pandas DataFrame.
    
    Parameters:
    - db_name (str): The path to the SQLite database file.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the tweet_id, tweet_type, user_id, and text of tweets
      that have received replies.
    """
    conn = create_connection(db_name)  # Establish a connection to the database
    query = """
    SELECT tweet_id, tweet_type, user_id, text
    FROM tweets
    WHERE tweet_id in (SELECT reply_to_status_id FROM replies) AND tweet_type = 'original'
    """  # SQL query to find original tweets that have replies
    df = run_query(query, conn)  # Execute the query and return the results as a DataFrame
    return df

def get_replies(db_name):
    """
    Retrieves all replies from the database along with their associated user IDs, reply IDs, and the IDs of the tweets they are replying to.
    
    This function establishes a connection to the SQLite database specified by the db_name parameter. It then executes a SQL query to select
    the user ID, reply ID, reply to status ID, and reply to user ID from the replies table, joining it with the tweets table to ensure that
    only replies that correspond to actual tweets are retrieved. The results are returned as a pandas DataFrame.
    
    Parameters:
    - db_name (str): The path to the SQLite database file from which to retrieve the replies.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the columns user_id, reply_id, reply_to_status_id, and reply_to_user_id for each reply in the database.
    """
    conn = create_connection(db_name)  # Establish a connection to the database
    query = """
    SELECT user_id, reply_id, reply_to_status_id, reply_to_user_id
    FROM replies
    INNER JOIN tweets ON replies.reply_id = tweets.tweet_id
    """  # SQL query to retrieve replies and their details
    df = run_query(query, conn)  # Execute the query and return the results as a DataFrame
    return df

def find_conv_2(user_id, check_id, replies):
    """
    Constructs a conversation chain starting from a specific tweet using a DataFrame of replies.

    This function iterates through the replies DataFrame to find all replies to a given tweet (identified by check_id)
    and recursively finds replies to those replies, building a nested dictionary structure that represents the entire
    conversation chain. Each key in the dictionary is a tuple of (user_id, tweet_id), and the value is a list of tuples,
    each containing 'user_id' and 'reply_id' of the replies.

    Parameters:
    - user_id (int): The user ID of the original tweet's author.
    - check_id (int): The tweet ID of the original tweet to start the conversation chain from.
    - replies (pandas.DataFrame): A DataFrame containing all replies, where each row represents a reply with columns
      for 'user_id', 'reply_id', and 'reply_to_status_id'.

    Returns:
    - dict: A dictionary representing the conversation chain, where each key is a tuple of (user_id, tweet_id),
      and each value is a list of tuples with 'user_id' and 'reply_id' of the replies.
    """
    this_replies = replies[replies['reply_to_status_id']==check_id]
    past_replies_dict={}
    now_reply_ids = list(this_replies['reply_id'])
    past_replies_dict[(user_id, check_id)] = list(this_replies[['user_id', 'reply_id']].itertuples(index=False, name=None))
    while len(now_reply_ids) != 0:
        past_replies_dict, now_reply_ids = check_for_replies_2(past_replies_dict,now_reply_ids, replies)
    return past_replies_dict

def check_for_replies_2(past_replies_dict, new_replies_list, replies):
    """
    Updates the conversation dictionary with replies to the tweets in the new_replies_list using a DataFrame.

    This function iterates over each tweet ID in new_replies_list, finds all replies to it in the replies DataFrame,
    and updates past_replies_dict with these new replies. Each key in past_replies_dict is a tuple of (reply_to_user_id, reply_to_status_id),
    and its value is a list of tuples, each containing 'user_id' and 'reply_id' of the replies. The function returns the updated
    dictionary and a list of new reply IDs to facilitate further recursive searches for replies.

    Parameters:
    - past_replies_dict (dict): The current dictionary of tweet conversations, where each key is a tuple
      of (reply_to_user_id, reply_to_status_id) and each value is a list of tuples containing 'user_id'
      and 'reply_id' of the replies.
    - new_replies_list (list): A list of tweet IDs to find replies for.
    - replies (pandas.DataFrame): A DataFrame containing all replies, where each row represents a reply with columns
      for 'user_id', 'reply_id', and 'reply_to_status_id'.

    Returns:
    - tuple: A tuple containing the updated past_replies_dict and a list of new reply IDs (now_reply_ids).
    """
    now_reply_ids = []
    for a_id in new_replies_list:
        this_replies = replies[replies['reply_to_status_id'] == a_id]
        returned_ids = list(this_replies['reply_id'])
        now_reply_ids += returned_ids
        if len(this_replies) != 0:
            past_replies_dict[list(this_replies[['reply_to_user_id', 'reply_to_status_id']].itertuples(index=False, name=None))[0]] = list(this_replies[['user_id', 'reply_id']].itertuples(index=False, name=None))
    return past_replies_dict, now_reply_ids

def build_tree(data):
    tree = Tree()
    tree.create_node(list(data.keys())[0], list(data.keys())[0])
    for key in list(data.keys()):
        for value in data[key]:
            ...
            tree.create_node(value,  value, parent = key)                                            
    return tree
                        
                
def get_all_replies_trees(tweets_with_replies, replies, db_name):
    """
    Constructs a dictionary of all replies trees for tweets that have received replies.

    This function iterates through each tweet that has received replies, as indicated by the
    tweets_with_replies DataFrame. For each tweet, it constructs a conversation tree starting
    from the tweet using the find_conv_2 function and the replies DataFrame. Each tree is stored
    in a dictionary with the tweet_id as the key.

    Parameters:
    - tweets_with_replies (pandas.DataFrame): A DataFrame containing tweets that have received replies.
      Each row must include 'tweet_id' and 'user_id'.
    - replies (pandas.DataFrame): A DataFrame containing all replies. This is used by the find_conv_2
      function to construct the conversation trees.
    - db_name (str): The path to the SQLite database file. This parameter is currently not used in the
      function but is included for potential future use or consistency with other function signatures.

    Returns:
    - dict: A dictionary where each key is a tweet_id and each value is a dictionary representing the
      conversation tree starting from that tweet. The conversation tree is constructed using the
      find_conv_2 function.
    """
    replies_trees = {}
    for n, (tweet_id, user_id) in enumerate(list(tweets_with_replies[['tweet_id', 'user_id']].itertuples(index=False, name=None))):
        if n % 100 == 0:
            print(n / 100)  # Progress indicator, prints every 100 iterations
        replies_trees[tweet_id] = find_conv_2(user_id, tweet_id, replies)
    return replies_trees

def get_all_replies_trees_from_csv(trees_file):
    """
    Reads a CSV file containing conversation trees and converts it into a dictionary.

    This function opens a CSV file specified by the `trees_file` parameter. Each line in the CSV file
    represents a conversation tree with the first element being the tweet ID and the subsequent elements
    being the serialized form of the conversation tree. The function deserializes these elements into a
    dictionary where each key is the tweet ID (converted to an integer) and the value is the conversation
    tree represented as a dictionary.

    Parameters:
    - trees_file (str): The path to the CSV file containing the serialized conversation trees.

    Returns:
    - dict: A dictionary where each key is a tweet ID and each value is the corresponding conversation tree
      as a dictionary. This structure allows for easy access and manipulation of conversation trees based on
      tweet IDs.
    """
    replies_trees={}
    with open(trees_file,'r') as data:
        for line in csv.reader(data):
            replies_trees[int(line[0])] = ast.literal_eval(", ".join(line[1:]))
    return replies_trees

def get_all_conversations(replies_trees, airlines):
    """
    Constructs and returns all possible conversations from a dictionary of replies trees for tweets that have received replies,
    filtering conversations that involve airlines.

    This function iterates through each replies tree in the replies_trees dictionary. For each tree, it constructs a conversation
    tree using the build_tree function and then finds all conversations within that tree that involve airlines using the
    find_conversations_in_a_tree function. The function aggregates all such conversations across all replies trees.

    Parameters:
    - replies_trees (dict): A dictionary where each key is a tweet_id and each value is a dictionary representing the
      conversation tree starting from that tweet. The conversation tree is constructed using the find_conv_2 function.
    - airlines (list): A list of airline user IDs. Conversations involving these user IDs are specifically targeted for extraction.

    Returns:
    - list: A list of all conversations involving airlines. Each conversation is represented as a list of tuples, where each tuple
      contains the user_id and tweet_id of a tweet in the conversation.

    Note:
    - The function prints a progress bar to the console to indicate the progress of processing the replies trees.
    """
    all_conversations =[]
    last=0
    divisor = len(replies_trees.keys())//100
    print("|"*100)
    for n,key in enumerate(replies_trees.keys()):
        current = n//divisor
        if current>last:
            print('|', end='')
        last=current
        built_tree = build_tree(replies_trees[key])
        all_conversations += find_conversations_in_a_tree(built_tree, airlines)
    return all_conversations

def find_conversations_in_a_tree(tree, airlines):
    """
    Identifies and extracts conversations involving airlines from a conversation tree.

    This function traverses each path from the root to the leaves of the provided conversation tree. It identifies
    conversations that involve airlines by checking if any node (tweet) in the path belongs to an airline. If a conversation
    between an airline and another user is found, it is added to a list of conversations. The function ensures that each
    conversation is only added once and checks for sub-conversations to avoid duplicates.

    Parameters:
    - tree (Tree): A Tree object representing the conversation tree to be analyzed. Each node in the tree should represent
      a tweet, identified by a tuple containing the user_id and tweet_id.
    - airlines (list): A list of integers representing the user IDs of airlines. Conversations involving these user IDs are
      targeted for extraction.

    Returns:
    - list: A list of conversations, where each conversation is represented as a list of tuples. Each tuple contains the
      user_id and tweet_id of a tweet in the conversation. The function also ensures that sub-conversations are not included
      in the final list by calling the check_for_sublists function.

    Note:
    - The function assumes that the tree is constructed in such a way that each path from the root to a leaf represents a
      complete conversation thread.
    - Conversations are identified based on the involvement of an airline user ID at any point in the conversation path.
    """
    conv = []  # Initialize a list to keep track of the current conversation's participants (airline and other user)
    tweets = []  # Initialize a list to collect tweets that are part of conversations involving airlines
    for path in tree.paths_to_leaves():  # Iterate over all paths from root to leaves in the tree
        tweet = []  # Initialize a list to collect tweets for the current path
        for n, node in enumerate(path):  # Iterate over each node (tweet) in the current path
            # Check if the current conversation list is empty and if the current or previous node involves an airline
            if len(conv) == 0:
                if node[0] in airlines and n != 0 and path[n-1][0] not in airlines:
                    # If the current node is an airline and the previous node is not, add both to the conversation
                    conv.append(node[0])
                    conv.append(path[n-1][0])
                    tweet.append(path[n-1])
                    tweet.append(node)
                elif node[0] in airlines and n == 0 and path[1][0] not in airlines:
                    # If the first node is an airline and the second node is not, add both to the conversation
                    conv.append(node[0])
                    conv.append(path[1][0])
                    tweet.append(node)
            elif len(conv) == 2:
                # If the conversation list has two participants, check if the current node belongs to one of them
                if node[0] in conv and node[0] != path[n-1][0]:
                    tweet.append(node)
                else:
                    # If the current node does not belong to the conversation participants, reset the conversation list
                    conv = []
                    if tweet not in tweets and tweet != []:
                        tweets.append(tweet)
                    tweet = []
        # After finishing the path, check if the current tweet list is unique and not empty, then add it to the tweets list
        if tweet not in tweets and tweet != []:
            tweets.append(tweet)
        tweet = []  # Reset the tweet list for the next path
        conv = []  # Reset the conversation participants list for the next path
    return check_for_sublists(tweets)  # Return the list of unique conversations, excluding sub-conversations            

def check_for_sublists(all_convs):
    """
    Removes sub-conversations from a list of conversations.

    This function iterates through a list of conversations, identifying and removing any sub-conversations
    that are fully contained within another conversation in the list. A sub-conversation is defined as a
    conversation where all its tweets are also part of another, larger conversation. This ensures that only
    unique, maximal conversations are retained in the list.

    Parameters:
    - all_convs (list of list of tuples): A list where each element is a conversation represented as a list of tuples.
      Each tuple contains the user_id and tweet_id of a tweet in the conversation.

    Returns:
    - list of list of tuples: The modified list of conversations with sub-conversations removed. Each conversation
      is represented as a list of tuples, where each tuple contains the user_id and tweet_id of a tweet in the
      conversation.
    """
    lists_to_rm=[]  # Initialize a list to keep track of sub-conversations to be removed
    for conv in all_convs:  # Iterate over all conversations
        for sub_conv in all_convs:  # For each conversation, iterate over all conversations again to find sub-conversations
            # Check if sub_conv is a sub-conversation of conv (all tweets in sub_conv are also in conv) and they are not the same conversation
            if(all(x in conv for x in sub_conv)) and conv!=sub_conv:
                if sub_conv not in lists_to_rm:  # If sub_conv is not already marked for removal, add it to the list
                    lists_to_rm.append(sub_conv)
    
    for a_list in lists_to_rm:  # Iterate over the list of sub-conversations to be removed
        all_convs.remove(a_list)  # Remove each sub-conversation from the original list of conversations
    return all_convs  # Return the modified list of conversations with sub-conversations removed

def get_list_of_conversations_with_text(all_conversations, db_name, airlines):
    """
    Retrieves the text and timestamp of tweets involved in all conversations, enriching each tweet with additional details.

    This function iterates through each conversation in the provided list, querying the database for the text and timestamp
    of each tweet involved in the conversation. It constructs a list of dictionaries, where each dictionary represents a tweet
    enriched with its text, timestamp, and the ID of the airline involved in the conversation. This allows for a detailed
    analysis of the conversations, including the content and timing of each exchange.

    Parameters:
    - all_conversations (list): A list of conversations, where each conversation is represented as a list of tuples.
      Each tuple contains the user_id and tweet_id of a tweet in the conversation.
    - db_name (str): The path to the SQLite database file from which to retrieve the tweet details.
    - airlines (list): A list of airline user IDs. This is used to identify which participant in the conversation is the airline.

    Returns:
    - list: A list of dictionaries, where each dictionary contains the conversation ID ('conv_id'), user ID ('user_id'),
      tweet ID ('tweet_id'), text of the tweet ('text'), timestamp of the tweet ('timestamp'), and the airline ID involved
      in the conversation ('for_airline_id').

    Note:
    - The function prints a progress bar to the console to indicate the progress of processing the conversations.
    - The function assumes that each conversation involves exactly one airline and at least one other user.
    """
    all_conversations_list_of_dicts = []  # Initialize an empty list to store the enriched tweet details
    conn = create_connection(db_name)  # Establish a connection to the database
    cursor = conn.cursor()  # Create a cursor object to execute SQL queries
    last = 0  # Variable to track the last progress update
    divisor = len(all_conversations) // 100  # Calculate the divisor for the progress bar
    print("|" * 100)  # Print the initial progress bar
    for conv_id, conv in enumerate(all_conversations):  # Iterate over each conversation
        current = conv_id // divisor  # Calculate the current progress
        if current > last:  # Check if it's time to update the progress bar
            print('|', end='')  # Print a progress update
        last = current  # Update the last progress tracker
        ids = list(set(i[0] for i in conv))  # Extract unique user IDs from the conversation
        if ids[0] in airlines:  # Determine which user ID belongs to an airline
            airline_id = ids[0]
        else:
            airline_id = ids[1]
        for tweet in conv:  # Iterate over each tweet in the conversation
            query = f'''
            SELECT text, timestamp_ms
            FROM tweets
            WHERE tweet_id = {tweet[1]}
            '''  # SQL query to retrieve the text and timestamp of the tweet
            cursor.execute(query)  # Execute the SQL query
            result = cursor.fetchone()  # Fetch the result of the query
            text = result[0]  # Extract the text of the tweet
            timestamp = result[1]  # Extract the timestamp of the tweet
            user_id = tweet[0]  # Extract the user ID from the tweet tuple
            tweet_id = tweet[1]  # Extract the tweet ID from the tweet tuple
            # Append a dictionary with the enriched tweet details to the list
            all_conversations_list_of_dicts.append({'conv_id': conv_id, 'user_id': user_id, 'tweet_id': tweet_id, 'text': text, 'timestamp': timestamp, 'for_airline_id': airline_id})
    conn.commit()  # Commit any changes to the database
    conn.close()  # Close the database connection
    return all_conversations_list_of_dicts  # Return the list of enriched tweet details

def get_list_of_conversations_from_file(file_name):
  """
  Reads a JSON file and returns its content as a Python object.

  This function is designed to read a file containing serialized JSON data, typically representing a list of conversations
  extracted from a social media platform. The JSON file is deserialized into a Python object (usually a list or a dictionary)
  for further processing or analysis.

  Parameters:
  - file_name (str): The path to the JSON file to be read.

  Returns:
  - result (list or dict): The Python object resulting from the deserialization of the JSON file's content. The exact type
    (list or dict) depends on the structure of the JSON data.
  """
  with open(file_name, 'r') as f:
      result = json.load(f)
  return result

replies_trees = get_all_replies_trees_from_csv(trees_file)
all_conversations_list_of_dicts = get_list_of_conversations_from_file(convs_file)

import json
import os
import sqlite3
import datetime

import country_converter as coco
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from sqlite3 import Error
from statistics import mean

db_name = '/Users/alexraudvee/Downloads/tweets_final.db'
convs_path = '/Users/alexraudvee/Downloads/updated_convs.json'

# path to the jsons with easyjet tweets and british airways tweets
easyjet_path = '/Users/alexraudvee/Downloads/easyjet_tweets.json'
british_path = '/Users/alexraudvee/Downloads/britishairways_tweets.json'

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


countries ={'AW': 'ABW',
 'AF': 'AFG',
 'AO': 'AGO',
 'AI': 'AIA',
 'AX': 'ALA',
 'AL': 'ALB',
 'AD': 'AND',
 'AE': 'ARE',
 'AR': 'ARG',
 'AM': 'ARM',
 'AS': 'ASM',
 'AQ': 'ATA',
 'TF': 'ATF',
 'AG': 'ATG',
 'AU': 'AUS',
 'AT': 'AUT',
 'AZ': 'AZE',
 'BI': 'BDI',
 'BE': 'BEL',
 'BJ': 'BEN',
 'BQ': 'BES',
 'BF': 'BFA',
 'BD': 'BGD',
 'BG': 'BGR',
 'BH': 'BHR',
 'BS': 'BHS',
 'BA': 'BIH',
 'BL': 'BLM',
 'BY': 'BLR',
 'BZ': 'BLZ',
 'BM': 'BMU',
 'BO': 'BOL',
 'BR': 'BRA',
 'BB': 'BRB',
 'BN': 'BRN',
 'BT': 'BTN',
 'BV': 'BVT',
 'BW': 'BWA',
 'CF': 'CAF',
 'CA': 'CAN',
 'CC': 'CCK',
 'CH': 'CHE',
 'CL': 'CHL',
 'CN': 'CHN',
 'CI': 'CIV',
 'CM': 'CMR',
 'CD': 'COD',
 'CG': 'COG',
 'CK': 'COK',
 'CO': 'COL',
 'KM': 'COM',
 'CV': 'CPV',
 'CR': 'CRI',
 'CU': 'CUB',
 'CW': 'CUW',
 'CX': 'CXR',
 'KY': 'CYM',
 'CY': 'CYP',
 'CZ': 'CZE',
 'DE': 'DEU',
 'DJ': 'DJI',
 'DM': 'DMA',
 'DK': 'DNK',
 'DO': 'DOM',
 'DZ': 'DZA',
 'EC': 'ECU',
 'EG': 'EGY',
 'ER': 'ERI',
 'EH': 'ESH',
 'ES': 'ESP',
 'EE': 'EST',
 'ET': 'ETH',
 'FI': 'FIN',
 'FJ': 'FJI',
 'FK': 'FLK',
 'FR': 'FRA',
 'FO': 'FRO',
 'FM': 'FSM',
 'GA': 'GAB',
 'GB': 'GBR',
 'GE': 'GEO',
 'GG': 'GGY',
 'GH': 'GHA',
 'GI': 'GIB',
 'GN': 'GIN',
 'GP': 'GLP',
 'GM': 'GMB',
 'GW': 'GNB',
 'GQ': 'GNQ',
 'GR': 'GRC',
 'GD': 'GRD',
 'GL': 'GRL',
 'GT': 'GTM',
 'GF': 'GUF',
 'GU': 'GUM',
 'GY': 'GUY',
 'HK': 'HKG',
 'HM': 'HMD',
 'HN': 'HND',
 'HR': 'HRV',
 'HT': 'HTI',
 'HU': 'HUN',
 'ID': 'IDN',
 'IM': 'IMN',
 'IN': 'IND',
 'IO': 'IOT',
 'IE': 'IRL',
 'IR': 'IRN',
 'IQ': 'IRQ',
 'IS': 'ISL',
 'IL': 'ISR',
 'IT': 'ITA',
 'JM': 'JAM',
 'JE': 'JEY',
 'JO': 'JOR',
 'JP': 'JPN',
 'KZ': 'KAZ',
 'KE': 'KEN',
 'KG': 'KGZ',
 'KH': 'KHM',
 'KI': 'KIR',
 'KN': 'KNA',
 'KR': 'KOR',
 'KW': 'KWT',
 'LA': 'LAO',
 'LB': 'LBN',
 'LR': 'LBR',
 'LY': 'LBY',
 'LC': 'LCA',
 'LI': 'LIE',
 'LK': 'LKA',
 'LS': 'LSO',
 'LT': 'LTU',
 'LU': 'LUX',
 'LV': 'LVA',
 'MO': 'MAC',
 'MF': 'MAF',
 'MA': 'MAR',
 'MC': 'MCO',
 'MD': 'MDA',
 'MG': 'MDG',
 'MV': 'MDV',
 'MX': 'MEX',
 'MH': 'MHL',
 'MK': 'MKD',
 'ML': 'MLI',
 'MT': 'MLT',
 'MM': 'MMR',
 'ME': 'MNE',
 'MN': 'MNG',
 'MP': 'MNP',
 'MZ': 'MOZ',
 'MR': 'MRT',
 'MS': 'MSR',
 'MQ': 'MTQ',
 'MU': 'MUS',
 'MW': 'MWI',
 'MY': 'MYS',
 'YT': 'MYT',
 'NA': 'NAM',
 'NC': 'NCL',
 'NE': 'NER',
 'NF': 'NFK',
 'NG': 'NGA',
 'NI': 'NIC',
 'NU': 'NIU',
 'NL': 'NLD',
 'NO': 'NOR',
 'NP': 'NPL',
 'NR': 'NRU',
 'NZ': 'NZL',
 'OM': 'OMN',
 'PK': 'PAK',
 'PA': 'PAN',
 'PN': 'PCN',
 'PE': 'PER',
 'PH': 'PHL',
 'PW': 'PLW',
 'PG': 'PNG',
 'PL': 'POL',
 'PR': 'PRI',
 'KP': 'PRK',
 'PT': 'PRT',
 'PY': 'PRY',
 'PS': 'PSE',
 'PF': 'PYF',
 'QA': 'QAT',
 'RE': 'REU',
 'RO': 'ROU',
 'RU': 'RUS',
 'RW': 'RWA',
 'SA': 'SAU',
 'SD': 'SDN',
 'SN': 'SEN',
 'SG': 'SGP',
 'GS': 'SGS',
 'SH': 'SHN',
 'SJ': 'SJM',
 'SB': 'SLB',
 'SL': 'SLE',
 'SV': 'SLV',
 'SM': 'SMR',
 'SO': 'SOM',
 'PM': 'SPM',
 'RS': 'SRB',
 'SS': 'SSD',
 'ST': 'STP',
 'SR': 'SUR',
 'SK': 'SVK',
 'SI': 'SVN',
 'SE': 'SWE',
 'SZ': 'SWZ',
 'SX': 'SXM',
 'SC': 'SYC',
 'SY': 'SYR',
 'TC': 'TCA',
 'TD': 'TCD',
 'TG': 'TGO',
 'TH': 'THA',
 'TJ': 'TJK',
 'TK': 'TKL',
 'TM': 'TKM',
 'TL': 'TLS',
 'TO': 'TON',
 'TT': 'TTO',
 'TN': 'TUN',
 'TR': 'TUR',
 'TV': 'TUV',
 'TW': 'TWN',
 'TZ': 'TZA',
 'UG': 'UGA',
 'UA': 'UKR',
 'UM': 'UMI',
 'UY': 'URY',
 'US': 'USA',
 'UZ': 'UZB',
 'VA': 'VAT',
 'VC': 'VCT',
 'VE': 'VEN',
 'VG': 'VGB',
 'VI': 'VIR',
 'VN': 'VNM',
 'VU': 'VUT',
 'WF': 'WLF',
 'WS': 'WSM',
 'YE': 'YEM',
 'ZA': 'ZAF',
 'ZM': 'ZMB',
 'ZW': 'ZWE'}

avia_companies = {'KLM': 56377143, 'AirFrance': 106062176, 'British_Airways': 18332190, 'AmericanAir': 22536055,
                  'Lufthansa': 124476322, 'AirBerlin': 26223583,
                  'AirBerlin assist': 2182373406, 'easyJet': 38676903, 'RyanAir': 1542862735, 'SingaporeAir': 253340062,
                  'Qantas': 218730857, 'EtihadAirways': 45621423,
                  'VirginAtlantic': 20626359}

exclude_list = avia_companies.values()

# functions that we are going to use

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn


def convert(label):
    dict = {'negative':-1, 'neutral':0, 'positive':1}
    return dict[label]


def get_margin(a3, df):
    try:
        margin = df[df['alpha_3'] == a3]['tweet_id'].values[0]
    except:
        margin = 0

    return margin


# create connection with the database 
conn = create_connection(db_name)
query1 = """
    SELECT tweet_geo_id AS tweet_id, country_code
    FROM tweets_geo
"""

tweet_geo_df = pd.read_sql_query(query1, conn)

# convert jsons to the dataframes 
df_for_easyjet = pd.read_json(easyjet_path)
df_for_britishairways = pd.read_json(british_path)

df_for_britishairways.drop(['timestamp_ms', 'text', 'sentiment_numeric', 'timestamp_int', 'text_clean_len', 'till_hour_str', 'day_of_year', 'till_minute_str'], axis=1, inplace=True)
df_for_easyjet.drop(['timestamp_ms', 'text', 'sentiment_numeric', 'timestamp_int', 'text_clean_len', 'till_hour_str', 'day_of_year', 'till_minute_str'], axis=1, inplace=True)
df_for_easyjet = pd.merge(df_for_easyjet, tweet_geo_df, on='tweet_id')
df_for_britishairways = pd.merge(df_for_britishairways, tweet_geo_df, on='tweet_id')

# take only negatives 
df_for_easyjet_neg = df_for_easyjet[df_for_easyjet['sentiment_label'] == 'negative']

# take only needed info from the dataframe for the plot 
df_neg_geo_ej = df_for_easyjet_neg[(df_for_easyjet_neg['sentiment_label'] == 'negative') ] 
df_neg_geo_ej.user_id.mask(df_neg_geo_ej.tweet_id.isin(exclude_list),inplace=True)

# group by country code with counted number of negatives tweets
df_neg_geo_ej = df_neg_geo_ej.groupby('country_code').count()[['tweet_id']].reset_index().drop(0)

# eliminate the row with XK because it is not in pandas anymore
df_neg_geo_ej = df_neg_geo_ej[df_neg_geo_ej.country_code != 'XK']

# add the alpha3 instead of the code 
df_neg_geo_ej['alpha_3'] = df_neg_geo_ej['country_code'].apply(lambda x: countries[x])

# df_neg_geo = df_neg_geo[df_neg_geo['tweet_id'] > 20]

# add all them in the world dataframe 
world['margin'] = world['iso_a3'].apply(lambda row: get_margin(row, df_neg_geo_ej))

world[world['margin'] != 0]
# create new color map 
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',['#D5F5E3', 'red', 'red'], N=256)

ax2 = world.plot(column='margin',  figsize=(16,16), edgecolor=u'gray', cmap=cmap)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.text(x=-190, y=-60, s="""Red - higher amount of negative tweets \nGreen - lower amount of negative tweets""", c='#34495E', size=12)
plt.axis(False)
plt.title('Negative sentiment about EasyJet', size=24)
plt.show()
plt.legend(['negative', ])
# take only negatives 
df_for_britishairways_neg = df_for_britishairways[df_for_britishairways['sentiment_label'] == 'negative']

# take only needed info from the dataframe for the plot 
df_neg_geo_ba = df_for_britishairways_neg[(df_for_britishairways_neg['sentiment_label'] == 'negative') ] 
df_neg_geo_ba.user_id.mask(df_neg_geo_ba.tweet_id.isin(exclude_list),inplace=True)

# group by country code with counted number of negatives tweets
df_neg_geo_ba = df_neg_geo_ba.groupby('country_code').count()[['tweet_id']].reset_index().drop(0)

# eliminate the row with XK because it is not in pandas anymore
df_neg_geo_ba = df_neg_geo_ba[df_neg_geo_ba.country_code != 'XK']

# add the alpha3 instead of the code 
df_neg_geo_ba['alpha_3'] = df_neg_geo_ba['country_code'].apply(lambda x: countries[x])

# add all them in the world dataframe 
world['margin'] = world['iso_a3'].apply(lambda row: get_margin(row, df_neg_geo_ba))

# create new color map 
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',['#D5F5E3', 'red', 'red'], N=256)

ax2 = world.plot(column='margin',  figsize=(16,16), edgecolor=u'gray', cmap=cmap)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.text(x=-190, y=-60, s="""Red - higher amount of negative tweets \nGreen - lower amount of negative tweets""", c='#34495E', size=12)
plt.axis(False)
plt.title('Negative sentiment about British Airways', size=24)
plt.show()
# get the top 5 countries with highest negative tweets as the list of country codes 
df_further_exploration_easy = df_neg_geo_ej.sort_values(by='tweet_id', ascending=False).head(5)
df_further_exploration_british = df_neg_geo_ba.sort_values(by='tweet_id', ascending=False).head(5)
top_country_with_negs = df_further_exploration_easy.country_code.to_frame().reset_index()
top_country_with_negs
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,8))

df_further_exploration_plt_easy = df_further_exploration_easy[['country_code', 'tweet_id']].set_index('country_code')
# plot
df_further_exploration_plt_easy.plot(kind='bar', ax=ax, color=['#E74C3C'])

# title
ax.text(x=0.12, y=.93, s="Top 5 countries with negative tweets about EasyJet", transform=fig.transFigure, ha='left', fontsize=24, weight='bold', alpha=.6)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

#legend 
ax.legend(['tweets count'])

# x axis 
ax.set_xlabel('countries')
ax.set_xticks(ticks=[0,1,2,3,4], labels=['Grate Britain', 'Spain', 'France', 'Italy', 'Germany'], rotation=30, size=14)

# y axis 
ax.set_ylabel('Tweets Count', size=14)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,8))

df_further_exploration_plt_british = df_further_exploration_british[['country_code', 'tweet_id']].set_index('country_code')
# plot
df_further_exploration_plt_british.plot(kind='bar', ax=ax, color=['#E74C3C'])

# title
ax.text(x=0.12, y=.93, s="Top 5 countries with negative tweets about British Airways", transform=fig.transFigure, ha='left', fontsize=24, weight='bold', alpha=.6)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

#legend 
ax.legend(['tweets count'])

# x axis 
ax.set_xlabel('countries')
ax.set_xticks(ticks=[0,1,2,3,4], labels=['Grate Britain', 'United States', 'Spain', 'South Africa', 'Italy'], rotation=30, size=14)

# y axis 
ax.set_ylabel('Tweets Count', size=14)
df_for_easyjet = pd.merge(df_for_easyjet, top_country_with_negs, on='country_code')
df_for_britishairways = pd.merge(df_for_britishairways, top_country_with_negs, on='country_code')

df_for_easyjet = pd.merge(df_for_easyjet, top_country_with_negs, on='country_code')
df_for_britishairways = pd.merge(df_for_britishairways, top_country_with_negs, on='country_code')
# create the lists with information by which we are going to extract about which topic is it

comforts = ['seat', 'legroom', 'hotel', 'greed', 'outpricing', 'charged', 'updates', 'website', 'flight experience', 'full price ticket', 'entertainment']
luggages = ['bad', 'luggage', 'baggage', 'handbag', 'belonging', 'suitcase', 'belongings']
punctuality = ['hour', 'day', 'late', 'waiting', 'delay', 'week', 'home', 'today', 'year', 'tomorrow', 'month', 'morning', 'date', 'night', 'minute', 'queue', 'cancel', 'cancelation', 'flight hour', 'flight tomorrow', 'filght day', 'flight  delay', 'date flight', 'hour delay', 'date flight', 'hour flight', 'return flight', 'flight home', 'flight today', 'week airline', 'flight delay', 'flight week', 'hold hour', 'day flight', 'today flight', 'time flight']
compensations = ['refund', 'money', 'time', 'compensation', 'hold', 'response', 'ticket', 'bill', 'price', 'return', 'refund flight', 'flight refund', 'refund option', 'refund week', 'website refund', 'passenger money', 'tiket fly', 'companion ticket', 'option passenger', 'passenger flight', 'compensation law', 'flight compensation', 'option refund']
customer_service = ['customer', 'service', 'payout', 'phone', 'help', 'someone', 'nothing', 'call', 'fleet', 'board', 'crew', 'ground', 'customer service', 'call centre', 'staff', 'support', 'airline website', 'phone line', 'ground fleet', 'manage booking']
foods = ['food', 'drink', 'snack']
flight_experiences = ['plane', 'flight', 'refund', 'hour', 'staff', 'money', 'support', 'customer', 'back', 'time', 'day', 'plane', 'delay', 'week', 'seat', 'option', 'need', 'compensation', 'company', 'baliouts', 'food', 'help', 'website', 'bag', 'travel', 'bood', 'luggage', 'booking', 'response', 'crew', 'people', 'passanger', 'cancellation', 'cancelation']
# function for finding the topics in the tweet 

def define_topic(text: str):

    text = text.lower() # convert to lower case 

    if any(word in text for word in comforts):
        return 'comfort'
    elif any(word in text for word in luggages):
        return 'luggage'
    elif any(word in text for word in punctuality):
        return 'punctuality'
    elif any(word in text for word in compensations):
        return 'compensation'
    elif any(word in text for word in customer_service):
        return 'customer services'
    elif any(word in text for word in foods):
        return 'food'
    elif any(word in text for word in flight_experiences):
        return 'flight experience'
    else:
        return None
# adding topics to the dataframe 

df_for_easyjet['topic'] = df_for_easyjet['text_clean'].apply(lambda row: define_topic(row)).dropna()
df_for_britishairways['topic'] = df_for_britishairways['text_clean'].apply(lambda row: define_topic(row)).dropna()
# function for transformation of the dataframe to need format for the plot 
def transformation(df):
    
    df_1 = df.pivot_table(values='tweet_id', index=df.topic, columns='sentiment_label')

    df_1['negative'] = df_1[['negative']]
    df_1['neutral'] = df_1[['neutral']].shift(-1)
    df_1['positive'] = df_1[['positive']].shift(-2)

    df_2 = df_1.dropna(how='all').fillna(0).sort_values(by='negative', ascending=False)
    
    return df_2
# apply function on different countries 

df_for_topics_british_GB = transformation(df_for_britishairways[df_for_britishairways['country_code'] == 'GB'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
df_for_topics_british_ES = transformation(df_for_britishairways[df_for_britishairways['country_code'] == 'ES'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
df_for_topics_british_FR = transformation(df_for_britishairways[df_for_britishairways['country_code'] == 'FR'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
df_for_topics_british_IT = transformation(df_for_britishairways[df_for_britishairways['country_code'] == 'IT'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
df_for_topics_british_DE = transformation(df_for_britishairways[df_for_britishairways['country_code'] == 'DE'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
# apply function on different countries 

df_for_topics_easy_GB = transformation(df_for_easyjet[df_for_easyjet['country_code'] == 'GB'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
df_for_topics_easy_ES = transformation(df_for_easyjet[df_for_easyjet['country_code'] == 'ES'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
df_for_topics_easy_FR = transformation(df_for_easyjet[df_for_easyjet['country_code'] == 'FR'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
df_for_topics_easy_IT = transformation(df_for_easyjet[df_for_easyjet['country_code'] == 'IT'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
df_for_topics_easy_DE = transformation(df_for_easyjet[df_for_easyjet['country_code'] == 'DE'].groupby(['topic', 'sentiment_label']).count()[['tweet_id']].reset_index())
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,7))

# plot
df_for_topics_easy_GB.plot(kind='bar', ax=ax, color=['#E74C3C', '#AAB7B8', '#2ECC71'])

# title
ax.text(x=0.12, y=.93, s="Frequent problems in Great Britain for EasyJet", transform=fig.transFigure, ha='left', fontsize=24, weight='bold', alpha=.6)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)


# x axis 
ax.set_xlabel('Topic of the tweets')
plt.xticks(rotation=30, size=14)

# y axis 
ax.set_ylabel('Tweets Count', size=14)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,7))

# plot
df_for_topics_easy_ES.plot(kind='bar', ax=ax, color=['#E74C3C', '#AAB7B8', '#2ECC71'])

# title
ax.text(x=0.12, y=.93, s="Frequent problems in Spain for EasyJet", transform=fig.transFigure, ha='left', fontsize=24, weight='bold', alpha=.6)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)


# x axis 
ax.set_xlabel('Topic of the tweets')
plt.xticks(rotation=30, size=14)

# y axis 
ax.set_ylabel('Tweets Count', size=14)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,7))

# plot
df_for_topics_easy_FR.plot(kind='bar', ax=ax, color=['#E74C3C', '#AAB7B8', '#2ECC71'])

# title
ax.text(x=0.12, y=.93, s="Frequent problems in France EasyJet", transform=fig.transFigure, ha='left', fontsize=24, weight='bold', alpha=.6)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)


# x axis 
ax.set_xlabel('Topic of the tweets')
plt.xticks(rotation=30, size=14)

# y axis 
ax.set_ylabel('Tweets Count', size=14)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,7))

# plot
df_for_topics_easy_IT.plot(kind='bar', ax=ax, color=['#E74C3C', '#AAB7B8', '#2ECC71'])

# title
ax.text(x=0.12, y=.93, s="Frequent problems in Italy for EasyJet", transform=fig.transFigure, ha='left', fontsize=24, weight='bold', alpha=.6)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)


# x axis 
ax.set_xlabel('Topic of the tweets')
plt.xticks(rotation=30, size=14)

# y axis 
ax.set_ylabel('Tweets Count', size=14)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,7))

# plot
df_for_topics_easy_DE.plot(kind='bar', ax=ax, color=['#E74C3C', '#AAB7B8', '#2ECC71'])

# title
ax.text(x=0.12, y=.93, s="Frequent problems in Germany for EasyJet", transform=fig.transFigure, ha='left', fontsize=24, weight='bold', alpha=.6)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)


# x axis 
ax.set_xlabel('Topic of the tweets')
plt.xticks(rotation=30, size=14)

# y axis 
ax.set_ylabel('Tweets Count', size=14)
df_easy_jet_all = pd.concat([df_for_topics_easy_DE, df_for_topics_easy_ES, df_for_topics_easy_FR, df_for_topics_easy_GB, df_for_topics_easy_IT]).groupby(['topic']).sum().sort_values(by='negative', ascending=False)
df_british_jet_all = pd.concat([df_for_topics_british_DE, df_for_topics_british_ES, df_for_topics_british_FR, df_for_topics_british_GB, df_for_topics_british_IT]).groupby(['topic']).sum().sort_values(by='negative', ascending=False)

# initialize the figure for the plots 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8), sharey=True)

# plots
bars_grouped1 = df_british_jet_all.plot(kind='bar', ax=ax[0], color=['#E74C3C', '#AAB7B8', '#2ECC71'])
bars_grouped2 = df_easy_jet_all.plot(kind='bar', ax=ax[1], color=['#E74C3C', '#AAB7B8', '#2ECC71'])

# title
fig.text(x=0.32, y=.93, s="Sentiment About Certain Topic", transform=fig.transFigure, ha='left', fontsize=28, weight='bold', alpha=.6)

fig.text(x=0.45, y=0, s="Problems with ...", transform=fig.transFigure, ha='left', fontsize=13, alpha=.9, weight='bold')

# set subtitle
ax[0].set_title('British Airways')
ax[1].set_title('EasyJet')

# Axis formatting.
ax[0].spines['top'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[0].spines['bottom'].set_color('#DDDDDD')
ax[1].spines['bottom'].set_color('#DDDDDD')
ax[0].set_axisbelow(True)
ax[1].set_axisbelow(True)
ax[0].yaxis.grid(True, color='#EEEEEE')
ax[1].yaxis.grid(True, color='#EEEEEE')
ax[0].xaxis.grid(False)
ax[1].xaxis.grid(False)


# x axis 
ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[0].tick_params(labelrotation=30)
ax[1].tick_params(labelrotation=30)

# y axis 
ax[0].set_ylabel('')
ax[1].set_ylabel('')
ax[0].set_ylabel('Number Of Tweets', size=14)

# saving code
# fig.get_figure().savefig('sentiment_about_topics.png')
def from_Timestamp_to_datetime_str(Timestamp):
    """
    the input should be: Timestamp(any)
    returns date as string: 'year-mont-day'
    """

    date_time = Timestamp.to_pydatetime().strftime('%Y-%m-%d')

    return date_time


def from_Timestamp_to_timestamp_int(Timestamp):
    """
    The input should be: Timestamp(any)
    Returns timestamp integer
    """

    timestamp_int = int(Timestamp.timestamp())

    return timestamp_int


def from_datetime_str_to_timestamp_int(date_str: str):
    """
    The input should be: 'year-mont-day'
    Returns timestamp integer
    """

    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    timestamp_int = int(date.timestamp())

    return timestamp_int
