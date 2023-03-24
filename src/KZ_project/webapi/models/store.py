from typing import Dict, List, Union
from database import db
from models.item import ItemJSON

StoreJSON = Dict[str, Union[int, str, List[ItemJSON]]]

class StoreModel(db.Model):    # tells SQLAlchemy that it is something that will be saved to database and will be retrieved from database

  __tablename__ = "stores"

  # Columns
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(80), unique = True)

  items = db.relationship("ItemModel", lazy="dynamic")  # this will allow us to check which item is in store whose id is equal to it's id.
                                      # lazy="dynamic" tells sqlalchemy to not create seperate objects for each item that is created
  def __init__(self, name: str):
    self.name = name

  def json(self) -> StoreJSON:
    return {
      "id": self.id, 
      "name": self.name, 
      "items": [item.json() for item in self.items.all()]
    }

  # searches the database for items using name
  @classmethod
  def find_store_by_name(cls, name: str) -> "StoreModel":
    # return cls.query.filter_by(name=name) # SELECT name FROM __tablename__ WHERE name=name
    # this function would return a StoreModel object
    return cls.query.filter_by(name=name).first() # SELECT name FROM __tablename__ WHERE name=name LIMIT 1

  @classmethod
  def find_all(cls) -> List["StoreModel"]:
    return cls.query.all()

  # method to insert or update an item into database
  def save_to_database(self) -> None:
    db.session.add(self)  # session here is a collection of objects that wil be written to database
    db.session.commit()

  def delete_from_database(self) -> None:
    db.session.delete(self)
    db.session.commit()