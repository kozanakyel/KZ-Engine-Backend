import datetime 
import pandas as pd
import os
import asyncio

## NOTEBOOK SDA CALISIYOR

from dotenv import load_dotenv

load_dotenv()

api_id=os.getenv('TG_api_id')
api_hash = os.getenv('TG_hash_id')
phone = os.getenv('TG_phone')
username = os.getenv('TG_username')

from telethon.sync import TelegramClient

client = TelegramClient(None, api_id, api_hash)

async def main():
    chats = ['investeniste']
    chat = 'Enisteinvest'
    df = pd.DataFrame()
    async for message in client.iter_messages(chat, limit=1000):
        # print(f'caht: {chat} sender: {message.sender_id} text: {message.text} date: {message.date}')
        data = {"group": [chat], "sender": [message.sender_id], "text": [message.text], "date": [message.date]}
        temp_df = pd.DataFrame(data)
        df = df.append(temp_df)

    df['date'] = df['date'].dt.tz_localize(None)  ## supheli tekrara bak
    print(df.head())
    df.to_csv('./burda.csv')

async with client:
    client.loop.run_until_complete(main())

import nest_asyncio
nest_asyncio.apply()