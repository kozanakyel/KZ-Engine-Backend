class Crypto:
    """
    Represents a cryptocurrency with its name, ticker symbol, and description.

    This class provides a way to store and retrieve information about a cryptocurrency.

    Attributes:
        name (str): The name of the cryptocurrency.
        ticker (str): The ticker symbol of the cryptocurrency.
        description (str): The description of the cryptocurrency.

    Methods:
        json(): Returns the cryptocurrency as a JSON-compatible dictionary.
    """

    def __init__(self, name: str, ticker: str, description: str):
        self.name = name.lower()
        self.ticker = ticker
        self.description = description

    def __eq__(self, other):
        """
        Compare if two cryptocurrencies are equal.

        Parameters:
            other (Crypto): The other cryptocurrency to compare.

        Returns:
            bool: True if the cryptocurrencies are equal, False otherwise.
        """
        if not isinstance(other, Crypto):
            return False
        return other.name == self.name

    def json(self):
        """
        Return the cryptocurrency as a JSON-compatible dictionary.

        Returns:
            dict: A dictionary representing the cryptocurrency.
        """
        return {
            'name': self.name,
            'ticker': self.ticker,
            'description': self.description
        }

    def __repr__(self):
        """
        Return a string representation of the cryptocurrency.

        Returns:
            str: A string representation of the cryptocurrency.
        """
        return f"<Crypto {self.name}>"
