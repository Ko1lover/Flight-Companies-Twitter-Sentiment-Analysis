# Sentiment Analysis on Twitter 

During this challenge I had worked with: https://github.com/Babinich, https://github.com/AlexRaudvee, https://github.com/timyuk
This challenge revolves around the question of how we compare the performance of airlines when they use Twitter as a communication channel. The dataset we are analyzing consists of a large number of tweets from several airlines, and we asked, in a role-playing game, to analyze the data for one of our clients. Our client is one out of several possible airlines, and in this project, the assigned airline is interested in assessing their performance when using Twitter as a communication channel. They are interested in comparing their performance to other airlines in general and an assigned competitor airline in particular. Our airline will be represented by an airline marketeer. The marketeer wonders whether their Twitter team is doing a good job and whether this is useful for the company, particularly in comparison to the competitor airline.

Our client is EasyJet company 

Our competitor is British Airways

## Table of Contents

- [Learning Goals of Project](#Learning-Goals-of-Project)
- [Data](#The-Data)
- [Files Description](#Files)
- [How To Run](#How-to-Run)


## Learning Goals of Project 

- independently apply and follow established data science research methods for a given problem and dataset,
- access, process, and reason about a large, complex dataset given in various data formats,
- independently find and familiarize themselves with programming languages, libraries, programs, and software packages for a specific purpose,
- implement a repeatable data analysis that makes use of existing libraries and programs in a self-chosen technical environment,
- independently validate results of their own and other student’s analyses using scientific techniques,
- solve a large data science task in a larger group of 6 students, and present their analysis and their findings in a presentation/poster suitable for a given audience.

## The Data
The data can be download from [this](https://surfdrive.surf.nl/files/index.php/s/Dz082kih8yMGB5P) link.
And you can find more details [here](documentation/Data_Description.pdf).

Given dataset represented in json files, later through our code it is stored in the database with SQLite

## Files

##### Data Exploration folder:
  - `data_collection_DB_creation.py` - collects the data from the json files does small preprocessing and stores the data in the database.
  - `get_conversations_geo.py` - from the database retrieves the conversations and in addition we retrieve the geo location of tweets if its available.
  - `get_reply_percentage.ipynb` - in this file we retrieve additional statistics which is going to be used in poster.

##### Sentiment Analysis folder:
  - `accuracy_of_the_model.ipynb` - in given file we estimate the accuracy of the model on our dataset.
  - `change_of_sentiment_in_conv.ipynb` - in given file we estimate the change of sentiment after company replies on the tweets of users.
  - `ONNX_SENTIMENT.ipynb` - given file represents our approach of applying sentiment extraction from tweets (we used ROBERTA model which is pre-trained on twitter data already)
  - `sentiment_analysis.ipynb` - in this file we provide more experiments and primary work with the model which extracts sentiment of the twits.
  - `sentiment_on_conversations_pre.ipynb` - this file contains the script for extracting sentiment of conversations.

##### Visualizations folder:

  - `difference_sen_resp_vis.ipynb` - in this file we represent the visualizations of how sentiment changes inside the conversations.
  - `main_visualizations.ipynb` - in this file we made the basic visualizations about data exploration part.
  - `poster_vis.ipynb` - this file contains all visualizations that were used in our poster.
  - `sentiment_analysis_vis.ipynb` - this file contains all visualizations about sentiment of tweets.
  - `sentiment_on_conversations_vis.ipynb` - this file contains all visualizations about sentiment inside the conversations and how it is changing.

##### Documentation folder: 
In given folder we store all the pdf files which can be used to dipper understand the data.

##### Main files
- `accuracy_check.json` - this file contains small sample of original dataset that were given.

- `config.py` - this is configuration file where you have to change the path to the data folders and ect.

## How to Run

##### For macOS and Linux:
1. Clone the repository:
`git clone https://github.com/AlexRaudvee/Sentiment_Analysis_Tiwitter` 
2. Navigate to the project directory:
`cd repository`

3. Create a virtual environment (optional but recommended):
`python3 -m venv venv`

4. Activate the virtual environment:
`source venv/bin/activate`

5. Install the required dependencies:
`pip install -r requirements.txt`
##### For Windows:

1. Clone the repository:
`git clone https://github.com/AlexRaudvee/Sentiment_Analysis_Tiwitter`
2. Navigate to the project directory:
`cd repository`
3. Create a virtual environment (optional but recommended):
`python -m venv venv`
4. Activate the virtual environment:
`.\venv\Scripts\activate`
5. Install the required dependencies:
`pip install -r requirements.txt`


##### If you do not want to clone the repository
In case if you struggle with cloning the repository, you can download the repository via the link above the code, after you did download the repository you can set up your environment with this same instructions as provided above starting from point 3.

