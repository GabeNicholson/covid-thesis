
import pandas as pd
import numpy as np
from covid_helper_functions import clean_dataframe, create_sentiment_per_day_df, get_article_grouped_df
pd.options.mode.string_storage = "pyarrow"
pd.options.mode.chained_assignment = None
health_data_df = pd.read_parquet('parquet/cleaned_health_data.parquet', engine='pyarrow')


def create_cached_dataframes(base_df_name, df_prefix, country_name, is_neutral_class=False, keep_top_n_publishers=False, us_covid=False):
    df = pd.read_parquet(f"parquet/{base_df_name}", engine='pyarrow')
    if keep_top_n_publishers:
        top_35_publishers = df['publisher'].value_counts()[:keep_top_n_publishers].index
        df = df[df.publisher.isin(top_35_publishers)]
        
    df = clean_dataframe(df, neutral_class=is_neutral_class)
    if us_covid:
        con_df = pd.read_parquet('parquet/cleaned_conservative_news_us.parquet', engine='pyarrow')
        df = pd.concat([df, con_df], axis=0, join='outer')
    df.to_parquet(f"cached_parquet/{df_prefix}_clean.parquet")
    
    grouped_df = get_article_grouped_df(df)
    grouped_df.to_parquet(f'cached_parquet/{df_prefix}_grouped.parquet')
    
    df_prod = create_sentiment_per_day_df(df, health_data_df.loc[country_name])
    df_prod.to_parquet(f'cached_parquet/{df_prefix}_prod.parquet')
    return df, grouped_df, df_prod


con_df = pd.read_parquet("conservative_news_predicted.parquet")
con_df = con_df.drop(columns=['Label'])
con_df['page_num'] = None
con_df['prediction'].replace({0: -1}, inplace=True)
con_df = con_df[con_df.date < '2022-05-01']
con_df['political_ideology'] = 'R'
con_df['national'] = True
con_df['Big_company'] = True
con_df.date = pd.to_datetime(con_df.date) # convert to datetime
con_df.date = con_df.date.dt.tz_localize(None) # get rid of timezone
con_df.date = con_df.date.dt.normalize() # get rid of hours
con_df = con_df[['date', 'publisher', 'is_oped', 'article_id', 'prediction', 'page_num','political_ideology', 'national', 'Big_company']]
con_df.to_parquet("parquet/cleaned_conservative_news_us.parquet")


# Covid News Predicted by COVID-SIEBERT
create_cached_dataframes("us_covid_news.parquet", "us", "United States", is_neutral_class=False, us_covid=True)
create_cached_dataframes("can_full_covid_readingweek.parquet", "can", "Canada", is_neutral_class=False)
create_cached_dataframes("eur_full_covid_readingweek-2.parquet", "eur", "United Kingdom", is_neutral_class=False)

# Regular News Predicted by COVID-SIEBERT
create_cached_dataframes("us-regular-bigfive-complete.parquet", "regular_us", "United States", is_neutral_class=False)
create_cached_dataframes("can_regular_full.parquet", "regular_can", "Canada", is_neutral_class=False, keep_top_n_publishers=40)
create_cached_dataframes("europe_regular_full.parquet", "regular_eur", "United Kingdom", is_neutral_class=False, keep_top_n_publishers=40);

# Covid Neutral News
create_cached_dataframes("us_full_covid_neutral.parquet", "us_neut", "United States", is_neutral_class=True)
create_cached_dataframes("eur_covid_neutral.parquet", "eur_neut", "United Kingdom", is_neutral_class=True)
create_cached_dataframes("can_covid_neutral.parquet", "can_neut", "Canada", is_neutral_class=True)

# Regular Neutral News
create_cached_dataframes("us_full_regular_neutral.parquet", "us_neut_regular", "United States", is_neutral_class=True)
create_cached_dataframes("eur_full_regular_neutral.parquet", "eur_neut_regular", "United Kingdom", is_neutral_class=True)
create_cached_dataframes("can_full_regular_neutral.parquet", "can_neut_regular", "Canada", is_neutral_class=True);


