from typing import Dict, List, Union
from KZ_project.webapi.database import db

TrackerJSON = Dict[str, Union[int, str, str, int]]

class TrackerCollection(db.Model):    # tells SQLAlchemy that it is something that will be saved to database and will be retrieved from database

  __tablename__ = "trackers"

  # Columns
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(80), unique= True)
  datetime_t = db.Column(db.String(100))
  position = db.Column(db.Integer())  # precision: numbers after decimal point

  asset_id = db.Column(db.Integer, db.ForeignKey("assets.id"))
  #store = db.relationship("StoreModel")

  def __init__(self, name: str, datetime_t: str, position: str, asset_id: int):
    self.name = name
    self.datetime_t = datetime_t
    self.position = position
    self.asset_id = asset_id

  def json(self) -> TrackerJSON:
    return {
      "id": self.id,
      "asset_id":self.asset_id,
      "name": self.name, 
      "position": self.position,
      "datetime_t": self.datetime_t
      }

  # searches the database for items using name
  @classmethod
  def find_item_by_name(cls, name: str) -> "TrackerCollection" :
    # return cls.query.filter_by(name=name) # SELECT name FROM __tablename__ WHERE name=name
    # this function would return a ItemModel object
    return cls.query.filter_by(name=name).first() # SELECT name FROM __tablename__ WHERE name=name LIMIT 1

  @classmethod
  def find_all(cls) -> List["TrackerCollection"]:
    return cls.query.all()

  # method to insert or update an item into database
  def save_to_database(self) -> None:
    db.session.add(self)  # session here is a collection of objects that wil be written to database
    db.session.commit()

  def delete_from_database(self) -> None:
    db.session.delete(self)
    db.session.commit()