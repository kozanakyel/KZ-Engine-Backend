import pandas as pd
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
from nltk.corpus import stopwords   # python -m nltk.downloader stopwords // from commandline

stop = stopwords.words('english')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re


class TweetSentimentAnalyzer():
    def __init__(self, df_twitter: pd.dataFrame()):
        self.df_tweets = df_twitter.copy()
        self.sid = SentimentIntensityAnalyzer()

    def cleaning_tweet_data(self, df: pd.DataFrame()):
        df_tweets = df.copy()
        df_tweets.dropna(inplace=True)
        if 'Unnamed: 0' in df_tweets.columns:
            df_tweets.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
        df_tweets.drop(columns=['source', 'name', 'location', 'verified', 'description'], axis=1, inplace=True)
        blanks = []  # start with an empty list

        for i, created_at, text, username in df_tweets.itertuples():  
            if type(text)==str:            
                if text.isspace():         
                    blanks.append(i)    

        df_tweets.drop(blanks, inplace=True)
        df_tweets = df_tweets.apply(lambda x: x.astype(str).str.lower()).drop_duplicates(subset=['text', 'username'], keep='first')

        df_tweets['text'] = df_tweets['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        df_tweets['text'] = df_tweets['text'].str.lower()
        df_tweets['text'] = df_tweets['text'].str.replace("@[a-z0-9A-Z]+", "", regex=True)
        df_tweets['text'] = df_tweets['text'].str.replace("#[a-z0-9A-Z]+","", regex=True)
        df_tweets['text'] = df_tweets['text'].str.replace(r"http\S+", "")
        df_tweets['text'] = df_tweets['text'].str.replace(r"www.\S+", "")
        df_tweets['text'] = df_tweets['text'].str.replace('[()!?]', ' ')
        df_tweets['text'] = df_tweets['text'].str.replace('\[.*?\]',' ')
        df_tweets['text'] = df_tweets['text'].str.replace("[^a-z0-9]"," ")
        df_tweets['words'] = df_tweets['text'].apply(lambda x:str(x.lower()).split())

        df_tweets['tweet_without_stopwords'] = df_tweets['text'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))

        def remove_stopword(x):
            return [y for y in x if y not in stopwords.words('english')]
        df_tweets['words'] = df_tweets['words'].apply(lambda x:remove_stopword(x))

        def remove_hashtag(x):
            return [y for y in x if not y.startswith('#')]
        df_tweets['words'] = df_tweets['words'].apply(lambda x:remove_hashtag(x))

        return df_tweets
    
    def preprocessing_tweet_datetime(self, df: pd.DataFrame()) -> pd.DataFrame():
        df_temp = df.copy()
        df_temp.drop(df_temp[df_temp.created_at == 'twitter'].index, inplace=True)
        df_temp.created_at = pd.to_datetime(df_temp.created_at)
        df_temp['Date'] = df_temp.created_at.apply(lambda x: x.date())
        df_temp['hour'] = df_temp.created_at.apply(lambda x: x.hour)
        df_temp['minute'] = df_temp.created_at.apply(lambda x: x.minute)
        return df_temp
    
    def get_vader_sentiment(self, score):    
        if (score >= 0.05): 
            return "Positive"
        elif (score < 0.05 and score > -0.05):
            return "Neutral"
        elif (score <= -0.05):    
            return "Negative"
        return score
    
    def get_sentiment_scores(self, df: pd.DataFrame()):
        df_temp = df.copy()
        df_temp['scores'] = df_temp['text'].apply(lambda review: self.sid.polarity_scores(review))
        df_temp['compound']  = df_temp['scores'].apply(lambda score_dict: score_dict['compound'])
        df_temp['comp_score'] = df_temp['compound'].apply(lambda score: self.get_vader_sentiment(score))
        return df_temp
    
    def get_most_common_words(self, df: pd.DataFrame(), number_of_words=20):
        df['words'] = df['tweet_without_stopwords'].apply(lambda x:str(x.lower()).split())
        top = Counter([item for sublist in df['words'] for item in sublist])
        temp = pd.DataFrame(top.most_common(number_of_words))
        temp.columns = ['Common_words','count']
        return temp

    def add_datetime_to_col(self, df: pd.DataFrame()) -> None:
        df['Datetime'] = np.nan
        for i in df.index:
            if df.hour[i] < 10:
                df.loc[i, 'Datetime'] = pd.to_datetime(f'{str(df.Date[i])} 0{str(df.hour[i])}:00+00:00')
            else:
                df.loc[i, 'Datetime'] = pd.to_datetime(f'{str(df.Date[i])} {str(df.hour[i])}:00+00:00')

    def get_sent_with_mean_interval(self, df_tweets: pd.DataFrame(), interval: str='1d') -> pd.DataFrame():
        df_result = pd.DataFrame()
        if interval == '1d':
            df_result['Date'] = df_tweets.Date.unique()
            df_result['compound_total'] = 0
            for i, dt, com in df_result.itertuples():  
                sub_df = df_tweets.loc[(df_tweets.comp_score != 'Neutral'), :]
                df_result.loc[i, 'compound_total'] = sub_df.loc[(sub_df.Datetime.unique()[i] == sub_df.Datetime), 'compound'].mean()
            df_result.set_index('Date', inplace=True)
        elif interval == '1h':
            df_result['Datetime'] = df_tweets.Datetime.unique()
            df_result['compound_total'] = 0
            for i, dt, com in df_result.itertuples():  
                sub_df = df_tweets.copy()
                df_result.loc[i, 'compound_total'] = sub_df.loc[(sub_df.Datetime.unique()[i] == sub_df.Datetime), 'compound'].mean()
            df_result.set_index('Datetime', inplace=True)
        df_result = df_result.sort_values('Datetime')
        return df_result
    
    def concat_ohlc_compound_score(self, ohlc: pd.DataFrame(), result_sent_df: pd.DataFrame()) -> pd.DataFrame():
        result = ohlc.merge(result_sent_df, how='outer', left_index=True, right_index=True)
        return result
    
    def plot_wordcloud(self, text, mask=None, max_words=200, max_font_size=50, figure_size=(16.0,9.0), color = 'white',
                   title = None, title_size=40, image_color=False):
        
        # plot_wordcloud(df_btc['tweet_without_stopwords'],mask=None,color='white',max_font_size=50,title_size=30,title="WordCloud of Bitcoin Tweets")
        stopwords = set(STOPWORDS)
        more_stopwords = {'u', "im"}
        stopwords = stopwords.union(more_stopwords)

        wordcloud = WordCloud(background_color=color,
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=400, 
                    height=400,
                    mask = mask)
        wordcloud.generate(str(text))
    
        plt.figure(figsize=figure_size)
        if image_color:
            image_colors = ImageColorGenerator(mask)
            plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
            plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
        else:
            plt.imshow(wordcloud)
            plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
        plt.axis('off')
        plt.tight_layout()  
