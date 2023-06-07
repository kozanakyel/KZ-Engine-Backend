class User:

    def __init__(self, wallet: str, username: str, email: str):
        self.wallet = wallet
        self.username = username
        self.email = email

    def __eq__(self, other):
        if not isinstance(other, User):
            return False
        return other.username == self.username

    def json(self):
        return {
            'wallet': self.wallet,
            'username': self.username,
            'email': self.email
        }

    def __repr__(self):
        return f"<User {self.username}>"
