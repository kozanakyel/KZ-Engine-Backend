from typing import Dict, List, Union
from KZ_project.webapi.database import db

AssetJSON = Dict[str, Union[int, str, float]]

class AssetCollection(db.Model):    # tells SQLAlchemy that it is something that will be saved to database and will be retrieved from database

  __tablename__ = "assets"

  # Columns
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(80), unique= True)
  price = db.Column(db.Float(precision=2))  # precision: numbers after decimal point

  aimodel_id = db.Column(db.Integer, db.ForeignKey("aimodels.id"))
  #store = db.relationship("StoreModel")

  def __init__(self, name: str, price: float, aimodel_id: int):
    self.name = name
    self.price = price
    self.aimodel_id = aimodel_id

  def json(self) -> AssetJSON:
    return {
      "id": self.id,
      "aimodel_id":self.aimodel_id,
      "name": self.name, 
      "price": self.price
      }

  # searches the database for items using name
  @classmethod
  def find_item_by_name(cls, name: str) -> "AssetCollection" :
    # return cls.query.filter_by(name=name) # SELECT name FROM __tablename__ WHERE name=name
    # this function would return a ItemModel object
    return cls.query.filter_by(name=name).first() # SELECT name FROM __tablename__ WHERE name=name LIMIT 1

  @classmethod
  def find_all(cls) -> List["AssetCollection"]:
    return cls.query.all()

  # method to insert or update an item into database
  def save_to_database(self) -> None:
    db.session.add(self)  # session here is a collection of objects that wil be written to database
    db.session.commit()

  def delete_from_database(self) -> None:
    db.session.delete(self)
    db.session.commit()