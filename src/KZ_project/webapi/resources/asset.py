from flask import Flask, request
from flask_restful import Resource, reqparse
from flask_jwt_extended import jwt_required, get_jwt, get_jwt_identity

from KZ_project.webapi.models.asset import AssetCollection

FIELD_BLANK_ERROR = "'{}' field cannot be left blank."
ITEM_NOT_FOUND = "Item not found! Please enter a valid item."
ITEM_NAME_ALREADY_EXISTS = "An item with name '{}' already exists!"
ERROR_INSERTING_ITEM = "Sorry! An error occured inserting the item!"
ADMIN_PRIVILEDGES_REQUIRED = "Admin privilages required for this action!"
ITEM_DELETED = "'{}' has been deleted successfully!"
LOGIN_TO_VIEW_DATA = "Please first login to view more data!"

asset_parser = reqparse.RequestParser()
asset_parser.add_argument("price",
    type=float,
    required=True,
    help= FIELD_BLANK_ERROR.format("price")
  )
asset_parser.add_argument("aimodel_id",
    type=int,
    required=True,
    help= FIELD_BLANK_ERROR.format("aimodel_id")
  )
asset_parser.add_argument("name",
    type=str,
    help= "name required for get"
  )

class Asset(Resource):
  
  # TO GET ITEM WITH NAME
  @classmethod
  @jwt_required()
  def get(cls):
    name = request.args.get("name")
    asset = AssetCollection.find_item_by_name(name)
    if asset:
      return asset.json(), 200
    
    return {"message": ITEM_NOT_FOUND}, 404

  # TO POST AN ITEM
  @classmethod
  @jwt_required(fresh=True)
  def post(cls):
    name = request.args.get("name")
    # if there already exists an item with "name", show a messege, and donot add the item
    if AssetCollection.find_item_by_name(name):
      return {"message": ITEM_NAME_ALREADY_EXISTS.format(name)} , 400

    data = asset_parser.parse_args()
    # data = request.get_json()   # get_json(force=True) means, we don't need a content type header
    asset = AssetCollection(**data)

    try:
      asset.save_to_database()
    except:
      return {"message": ERROR_INSERTING_ITEM}, 500
    
    return asset.json(), 201  # 201 is for CREATED status

  # TO DELETE AN ITEM
  @classmethod
  @jwt_required()
  def delete(cls, name: str):
    claims = get_jwt()

    if not claims["is_admin"]:
      return {"message": ADMIN_PRIVILEDGES_REQUIRED}, 401

    asset = AssetCollection.find_item_by_name(name)
    if asset:
      asset.delete_from_database()
      return {"message": ITEM_DELETED.format(name)}, 200

    # if doesn't exist, skip deleting 
    return {"message": ITEM_NOT_FOUND}, 404

  # TO ADD OR UPDATE AN ITEM
  @classmethod
  def put(cls, name: str):
    data = Asset.parser.parse_args()
    # data = request.get_json()
    asset = AssetCollection.find_item_by_name(name)

    # if item is not available, add it
    if asset is None:
      asset = AssetCollection(name, **data)
    # if item exists, update it
    else:
      asset.price = data['price']
      asset.aimodel_id = data['aimodel_id']
    
    # whether item is changed or inserted, it has to be saved to db
    asset.save_to_database()
    return asset.json()


class AssetList(Resource):
  @classmethod
  @jwt_required(optional=True)
  def get(cls):
    user_id = get_jwt_identity()
    assets = [asset.json() for asset in AssetCollection.find_all()]

    # if user id is given, then display full details
    if user_id:
      return {"assets": assets}, 200

    # else display only item name
    return {
      "assets": [asset["name"] for asset in assets],
      "message": LOGIN_TO_VIEW_DATA
    }, 200