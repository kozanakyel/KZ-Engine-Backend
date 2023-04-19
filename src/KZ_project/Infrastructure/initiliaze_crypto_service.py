from KZ_project.Infrastructure import config
from KZ_project.core.adapters.crypto_repository import CryptoRepository
from KZ_project.webapi.services import services
from KZ_project.Infrastructure.orm_mapper import orm

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

orm.start_mappers()
engine = create_engine(config.get_postgres_uri())
get_session = sessionmaker(bind=engine)
orm.metadata.create_all(engine)

coin_list = config.config_coin_list

def add_all_cryptos():
    for crypto in coin_list:
        session = get_session()
        repo = CryptoRepository(session)
        try:
            services.add_crypto(
                crypto.SYMBOL_NAME,
                crypto.SYMBOL_CUT,
                crypto.description,
                repo,
                session
            )
        except (services.InvalidName) as e:
            print(f"errors: {e}")
            
def create_all_models_2hourly():
    




