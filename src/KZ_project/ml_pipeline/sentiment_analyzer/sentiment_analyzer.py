from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('stopwords')
from nltk.corpus import (
    stopwords,
)  # python -m nltk.downloader stopwords // from commandline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import os
import warnings
from KZ_project.Infrastructure.logger.logger import Logger
from googletrans import Translator
from tqdm import tqdm

warnings.filterwarnings("ignore")
stop = stopwords.words("english")
tqdm.pandas()


class SentimentAnalyzer:
    def __init__(self, lang: str = "en", logger: Logger = None):
        # self.df_tweets = df_twitter.copy()
        self.lang = lang
        self.logger = logger
        self.sid = SentimentIntensityAnalyzer()

    def create_sentiment_scores(self, df_tweets: pd.DataFrame()) -> pd.DataFrame():
        df = self.cleaning_tweet_data(df_tweets)
        df = self.preprocessing_tweet_datetime(df)
        df = self.get_sentiment_scores(df)
        self.add_datetime_to_col(df)
        return df

    def cleaning_tweet_data(self, df: pd.DataFrame()):
        df_tweets = df.copy()
        if "Unnamed: 0" in df_tweets.columns:
            df_tweets.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
        if "source" in df_tweets.columns:
            df_tweets.drop(
                columns=["source", "name", "location", "verified", "description"],
                axis=1,
                inplace=True,
            )
            self.log(f"Extracted unnecassary columns for tweets")

        df_tweets = df_tweets.apply(
            lambda x: x.astype(str).str.lower()
        ).drop_duplicates(subset=["text", "username"], keep="first")
        self.log(f"Extracted Duplicates rows from tweets")

        df_tweets["text"] = df_tweets["text"].apply(
            lambda x: re.split("https:\/\/.*", str(x))[0]
        )
        df_tweets["text"] = df_tweets["text"].str.lower()
        df_tweets["text"] = df_tweets["text"].str.replace(
            "@[a-z0-9A-Z]+", "", regex=True
        )
        df_tweets["text"] = df_tweets["text"].str.replace(
            "#[a-z0-9A-Z]+", "", regex=True
        )

        blanks = []  # start with an empty list
        for i, created_at, text, *username in df_tweets.itertuples():
            if type(text) == str:
                if text.isspace():
                    blanks.append(i)
        df_tweets.drop(blanks, inplace=True)
        # df_tweets['text'] = df_tweets['text'].str.replace(r"http\S+", "")
        # df_tweets['text'] = df_tweets['text'].str.replace(r"www.\S+", "")
        # df_tweets['text'] = df_tweets['text'].str.replace('[()!?]', ' ')
        # df_tweets['text'] = df_tweets['text'].str.replace('\[.*?\]',' ')
        # df_tweets['text'] = df_tweets['text'].str.replace("[^a-z0-9]"," ")

        if self.lang == "en":
            df_tweets["tweet_without_stopwords"] = df_tweets["text"].apply(
                lambda x: " ".join(
                    [word for word in str(x).split() if word not in (stop)]
                )
            )
            df_tweets["words"] = df_tweets["text"].apply(
                lambda x: str(x.lower()).split()
            )
        df_tweets.dropna(inplace=True)
        self.log(f"Cleaning Tweets for blanks tweets and hastag username values")

        return df_tweets

    def preprocessing_tweet_datetime(self, df: pd.DataFrame()) -> pd.DataFrame():
        """
        For adding datetime groups Date, Hour, Minute to existing Dataframe.
            It uses to copy of existing Dataframe
        Args:
            self (tsa)
            df (DataFrame)

        Returns:
            DataFrame
        """
        df_temp = df.copy()
        # Fixed for some blank tweets: 'tweets'. Interesting api result delete below with function
        df_temp.drop(df_temp[df_temp.created_at == "twitter"].index, inplace=True)
        df_temp.created_at = pd.to_datetime(df_temp.created_at)
        df_temp["Date"] = df_temp.created_at.apply(lambda x: x.date())
        df_temp["hour"] = df_temp.created_at.apply(lambda x: x.hour)
        df_temp["minute"] = df_temp.created_at.apply(lambda x: x.minute)
        self.log(f"Added Datetime fatures for tweets to df (date, hour, minute)")
        return df_temp

    def get_vader_sentiment(self, score):
        """
        Return to float sentiment score to with Positive, Neurtal and Negative label
        Args:
            self (tsa)
            score (float)

        Returns: str
        """
        if score >= 0.05:
            return "Positive"
        elif 0.05 > score > -0.05:
            return "Neutral"
        elif score <= -0.05:
            return "Negative"
        return score

    def translate_text(self, text, src="tr", dest="en"):
        """
        For translate sentences Checking task the input is None or not None,
           Because mostly you can encounter the TypeError because of error inside the googletranslator
           bug.

        Args:
            text (str): text
            src (str, optional): language for source Defaults to 'tr'.
            dest (str, optional): language for destination. Defaults to 'en'.

        Returns:
            str: translated text
        """
        if text is None or text.strip() == "":
            return "Please provide a valid string to translate."

        translator = Translator()
        result = translator.translate(text, src=src, dest=dest)
        return result.text

    def get_sentiment_scores(self, df: pd.DataFrame()):
        df_temp = df.copy()
        if self.lang == "tr":
            self.log(f"Started Turkish to English translate Process")
            df_temp["text"] = df_temp["text"].progress_apply(
                self.translate_text, src="tr", dest="en"
            )

        df_temp["scores"] = df_temp["text"].apply(
            lambda review: self.sid.polarity_scores(review)
        )
        df_temp["compound"] = df_temp["scores"].apply(
            lambda score_dict: score_dict["compound"]
        )
        df_temp["comp_score"] = df_temp["compound"].apply(
            lambda score: self.get_vader_sentiment(score)
        )
        self.log(
            f"Created Sentiment Polarity Scores columns (comp_score, compound, scores)"
        )
        return df_temp

    def get_most_common_words(self, df: pd.DataFrame(), number_of_words=20):
        df["words"] = df["tweet_without_stopwords"].apply(
            lambda x: str(x.lower()).split()
        )
        top = Counter([item for sublist in df["words"] for item in sublist])
        temp = pd.DataFrame(top.most_common(number_of_words))
        temp.columns = ["Common_words", "count"]
        return temp

    def add_datetime_to_col(self, df: pd.DataFrame()) -> None:
        df["Datetime"] = np.nan
        for i in df.index:
            if df.hour[i] < 10:
                df.loc[i, "Datetime"] = pd.to_datetime(
                    f"{str(df.Date[i])} 0{str(df.hour[i])}:00+00:00"
                )
            else:
                df.loc[i, "Datetime"] = pd.to_datetime(
                    f"{str(df.Date[i])} {str(df.hour[i])}:00+00:00"
                )

    def get_sent_with_mean_interval(
        self, df_tweets: pd.DataFrame(), interval: str = "1h"
    ) -> pd.DataFrame():
        df_result = pd.DataFrame()
        df_result["Datetime"] = df_tweets.Datetime.unique()
        df_result["sentiment_score"] = 0
        for i, dt, com in df_result.itertuples():
            sub_df = df_tweets.copy()
            df_result.loc[i, "sentiment_score"] = sub_df.loc[
                (sub_df.Datetime.unique()[i] == sub_df.Datetime), "compound"
            ].mean()
        df_result.set_index("Datetime", inplace=True)
        df_result = df_result.sort_values("Datetime")
        if interval == "1d":
            df_result = df_result.resample("D").mean().reset_index()
            df_result.set_index("Datetime", inplace=True)
            df_result = df_result.sort_values("Datetime")
        df_result = df_result.dropna(axis=0)
        return df_result

    def concat_ohlc_compound_score(
        self, ohlc: pd.DataFrame(), result_sent_df: pd.DataFrame()
    ) -> pd.DataFrame():
        result = ohlc.merge(
            result_sent_df, how="inner", left_index=True, right_index=True
        )
        result["compound_total"] = result["compound_total"].fillna(0)
        self.log(f"Concatenation for sentiment score tweets and indicator Matrix data")
        return result

    def create_sent_results_df(self, df_tweets: pd.DataFrame()) -> tuple:
        df = self.create_sentiment_scores(df_tweets)

        df_result_day = self.get_sent_with_mean_interval(df, interval="1d")
        df_result_hour = self.get_sent_with_mean_interval(df, interval="1h")

        return df_result_day, df_result_hour

    def plot_wordcloud(
        self,
        text,
        mask=None,
        max_words=200,
        max_font_size=50,
        figure_size=(16.0, 9.0),
        color="white",
        title=None,
        title_size=40,
        image_color=False,
    ):
        # plot_wordcloud(df_btc['tweet_without_stopwords'],mask=None,color='white',max_font_size=50,title_size=30,
        # title="WordCloud of Bitcoin Tweets")
        stopwords = set(STOPWORDS)
        more_stopwords = {"u", "im"}
        stopwords = stopwords.union(more_stopwords)

        wordcloud = WordCloud(
            background_color=color,
            stopwords=stopwords,
            max_words=max_words,
            max_font_size=max_font_size,
            random_state=42,
            width=600,
            height=200,
            mask=mask,
        )
        wordcloud.generate(str(text))

        plt.figure(figsize=figure_size)
        if image_color:
            image_colors = ImageColorGenerator(mask)
            plt.imshow(
                wordcloud.recolor(color_func=image_colors), interpolation="bilinear"
            )
            plt.title(
                title, fontdict={"size": title_size, "verticalalignment": "bottom"}
            )
        else:
            plt.imshow(wordcloud)
            plt.title(
                title,
                fontdict={
                    "size": title_size,
                    "color": "black",
                    "verticalalignment": "bottom",
                },
            )
        plt.axis("off")
        plt.tight_layout()

    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
