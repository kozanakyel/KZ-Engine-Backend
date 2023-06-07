import abc
from KZ_project.core.adapters.repository import AbstractBaseRepository
from KZ_project.core.domain.user import User


class AbstractUserRepository(AbstractBaseRepository):
    @abc.abstractmethod
    def add(self, user: User):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, wallet) -> User:
        raise NotImplementedError

    @abc.abstractmethod
    def list(self):
        raise NotImplementedError


class UserRepository(AbstractUserRepository):
    def __init__(self, session):
        self.session = session

    def add(self, user):
        self.session.add(user)

    def get(self, wallet):
        return self.session.query(User).filter_by(wallet=wallet).first()

    def list(self):
        return self.session.query(User).all()
