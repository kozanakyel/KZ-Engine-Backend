import snscrape.modules.twitter as sntwitter
import pandas as pd

class SnsTwitterCollection:
    def __init__(self, logger=None, connection=True):
        
        self.logger = logger
        self.connect = connection

    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)

    def connect_twitter(self):
        # No need to connect with SNSscrape
        pass

    def get_tweets(self, search_query: str, lang: str, start_time: str, end_time: str):
        query = f"{search_query} lang:{lang} since:{start_time} until:{end_time}"
        tweets_list = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= 100:
                break
            tweets_list.append([tweet.date, tweet.content, tweet.user.username, tweet.user.location, tweet.user.verified, tweet.user.description])
        tweets_df = pd.DataFrame(tweets_list, columns=['created_at', 'Text', 'Username', 'Location', 'Verified', 'Description'])
        return tweets_df
    
if __name__ == '__main__':
    snstwt= SnsTwitterCollection()
    
    search_query = 'btc'
    lang = 'en'
    start_time = '2023-04-20'
    end_time = '2023-04-22'
    tweets_df = snstwt.get_tweets(search_query=search_query, lang=lang, start_time=start_time, end_time=end_time)
    print(f'info: {tweets_df.info()} head: {tweets_df.head()}')