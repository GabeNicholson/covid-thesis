import pandas as pd
import numpy as np
from tsmoothie.smoother import LowessSmoother

# political_ideology = (pd.read_csv('top_30_publishers_annotate_UPDATED.csv', index_col=[0]))
political_ideology = pd.read_parquet('prod_annotated_publishers.parquet')

# political_ideology = (pd.read_csv('top_30_publishers_annotate_UPDATED.csv')
#                     .rename(columns={'Unnamed: 0': 'publisher', 'publisher': 'drop_this'})
#                     .drop(columns=['drop_this', 'nuanced']))
    
def classify_article_class(article_mean_sentiment, larger_cutoff=0.1):
    if article_mean_sentiment > larger_cutoff:
        return 1
    elif article_mean_sentiment < -larger_cutoff:
        return -1
    else:
        return 0

def neutral_article_class(sentiment):
    if sentiment == 2:
        return 1
    elif sentiment == 1:
        return 0 # neutral
    else:
        return -1

def clean_dataframe(dataframe, neutral_class=False):
    """ Fixes publisher string formatting, prediction range to be between -1 and 1,
    and adds publisher political ideology.
    """
    if "bert_prediction" in dataframe.columns:
        dataframe['bert_prediction'].mean()
        dataframe['bert_prediction'] = dataframe['bert_prediction'].replace(0, -1)
        dataframe['bert_prediction'].mean()
        dataframe['vader_binary_prediction'] = np.where(dataframe['vader_prediction'] > 0, 1, -1) 
        dataframe['vader_nuance_prediction'] = dataframe['vader_prediction'].apply(lambda x: classify_article_class(x, 0.05))
        
    if neutral_class:
        dataframe['prediction'] = dataframe['prediction'].apply(lambda x: neutral_article_class(x))
    else:
        dataframe['prediction'] = np.where(dataframe['prediction'] == 0, -1, 1) # More intuitive sentiment score scales.
        dataframe = dataframe.groupby('article_id', as_index=False).nth([0,1,2,3,4]) # the last sentence block is strangely positive, so remove it.
        
    dataframe['publisher'] = dataframe['publisher'].str.replace('\(Online\)', '', regex=True).str.rstrip()
    dataframe['publisher'] = dataframe['publisher'].str.replace('Washington Post  The', 'Washington Post The', regex=False)
    dataframe['publisher'] = dataframe['publisher'].str.title()
    dataframe['publisher'] = dataframe['publisher'].apply(lambda x: ' '.join(x.split()))
    dataframe['publisher'] = dataframe['publisher'].replace('Toronto Star The', 'Toronto Star')
    dataframe['publisher'] = dataframe['publisher'].replace('Leader Post The', 'Leader Post')
    dataframe['publisher'] = dataframe['publisher'].replace('Independent (Daily Edition) The', 'Independent The')
    
    # combine online publications
    dataframe = dataframe.merge(political_ideology, on='publisher', how='left') # left join so we don't create NA rows for Canada and UK.
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['page_num'] = dataframe['page_num'].astype('string[pyarrow]')
    return dataframe

def text_class(x):
    if x == 1:
        return 'Positive'
    elif x == -1:
        return 'Negative'
    return 'Neutral'

def get_article_grouped_df(fresh_dataframe, agg_dict={'prediction':'mean', 'publisher':'first', 'page_num':'first', 'political_ideology':'first', 'Big_company':'first', 'national':'first'}):
    if 'vader_prediction' in fresh_dataframe.columns:
        agg_dict = agg_dict | {'vader_prediction':'mean', 'bert_prediction':'mean', 'vader_binary_prediction':'mean'}
    article_grouped_df = fresh_dataframe.groupby(['date', "article_id"], as_index=False).agg(agg_dict)
    article_grouped_df['article_classification'] = article_grouped_df['prediction'].apply(classify_article_class)
    article_grouped_df['article_classification_text'] = article_grouped_df['article_classification'].apply(text_class)
    return article_grouped_df

def create_sentiment_per_day_df(fresh_dataframe, health_dataframe=None):
    """First it groups by article. Then it groups by the date. If a health_dataframe is passed then the final
    dataframe is also merged with health statistics. Relevant columns are also smoothed so they can be plotted. 

    Args:
        fresh_dataframe (pandas dataframe): can be cleaned or not.
        health_dataframe (_type_, optional): Our World in Data health Dataframe. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with each row corresponding to a single day.
    """
    article_grouped_df = get_article_grouped_df(fresh_dataframe)
    AGG_FUNCS = {'prediction':'mean', 'article_classification':'mean', 'article_id':'nunique'}
    if 'vader_prediction' in article_grouped_df.columns:
        AGG_FUNCS = AGG_FUNCS | {'vader_prediction':'mean', 'bert_prediction':'mean', 'vader_binary_prediction':'mean'}
    sentiment_per_day_df = article_grouped_df.groupby('date', as_index=False).agg(AGG_FUNCS).rename(columns={'article_id': 'num_articles_published'})
    if type(health_dataframe) == pd.DataFrame:
        sentiment_per_day_df = sentiment_per_day_df.merge(health_dataframe, on='date', how='inner')
        sentiment_per_day_df['positive_rate_diff'] = sentiment_per_day_df['positive_rate'].diff()
        sentiment_per_day_df['new_cases_smoothed_diff'] = sentiment_per_day_df['new_cases_smoothed'].diff()
        sentiment_per_day_df['weekly_hosp_admissions_diff'] = sentiment_per_day_df['weekly_hosp_admissions'].diff()
        
    smoother = LowessSmoother(smooth_fraction=0.02, iterations=2)
    smoother.smooth((sentiment_per_day_df['prediction'], sentiment_per_day_df['article_classification'], sentiment_per_day_df['num_articles_published']))
    sentiment_per_day_df['smoothed_prediction'] = smoother.smooth_data[0]
    sentiment_per_day_df['smoothed_article_class'] = smoother.smooth_data[1]
    sentiment_per_day_df['Articles Published per Day'] = smoother.smooth_data[2]
    lower_bound_prediction, upper_bound_prediction = smoother.get_intervals('sigma_interval', n_sigma=1.5)
    sentiment_per_day_df['lower_bound_prediction'] = lower_bound_prediction[0]
    sentiment_per_day_df['upper_bound_prediction'] = upper_bound_prediction[0]
    sentiment_per_day_df['lower_bound_classification'] = lower_bound_prediction[1]
    sentiment_per_day_df['upper_bound_classification'] = upper_bound_prediction[1]
    return sentiment_per_day_df


def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


# test example satka.