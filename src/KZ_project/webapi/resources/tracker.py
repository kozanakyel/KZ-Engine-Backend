from flask import Flask, request
from flask_restful import Resource, reqparse
from flask_jwt_extended import jwt_required, get_jwt, get_jwt_identity

from KZ_project.webapi.models.tracker import TrackerCollection

FIELD_BLANK_ERROR = "'{}' field cannot be left blank."
ITEM_NOT_FOUND = "Item not found! Please enter a valid item."
ITEM_NAME_ALREADY_EXISTS = "An item with name '{}' already exists!"
ERROR_INSERTING_ITEM = "Sorry! An error occured inserting the item!"
ADMIN_PRIVILEDGES_REQUIRED = "Admin privilages required for this action!"
ITEM_DELETED = "'{}' has been deleted successfully!"
LOGIN_TO_VIEW_DATA = "Please first login to view more data!"

tracker_parser = reqparse.RequestParser()
tracker_parser.add_argument("position",
    type=int,
    required=True,
    help= FIELD_BLANK_ERROR.format("position")
  )
tracker_parser.add_argument("asset_id",
    type=int,
    required=True,
    help= FIELD_BLANK_ERROR.format("asset_id")
  )
tracker_parser.add_argument("name",
    type=str,
    help= "name required for get"
  )
tracker_parser.add_argument("datetime_t",
    type=str,
    help= "datetime_t required for get"
  )

class Tracker(Resource):
  
  # TO GET ITEM WITH NAME
  @classmethod
  @jwt_required()
  def get(cls):
    name = request.args.get("name")
    tracker = TrackerCollection.find_item_by_name(name)
    if tracker:
      return tracker.json(), 200
    
    return {"message": ITEM_NOT_FOUND}, 404

  # TO POST AN ITEM
  @classmethod
  @jwt_required(fresh=True)
  def post(cls):
    name = request.args.get("name")
    # if there already exists an item with "name", show a messege, and donot add the item
    if TrackerCollection.find_item_by_name(name):
      return {"message": ITEM_NAME_ALREADY_EXISTS.format(name)} , 400

    data = tracker_parser.parse_args()
    # data = request.get_json()   # get_json(force=True) means, we don't need a content type header
    tracker = TrackerCollection(**data)

    try:
      tracker.save_to_database()
    except:
      return {"message": ERROR_INSERTING_ITEM}, 500
    
    return tracker.json(), 201  # 201 is for CREATED status

  # TO DELETE AN ITEM
  @classmethod
  @jwt_required()
  def delete(cls, name: str):
    claims = get_jwt()

    if not claims["is_admin"]:
      return {"message": ADMIN_PRIVILEDGES_REQUIRED}, 401

    tracker = TrackerCollection.find_item_by_name(name)
    if tracker:
      tracker.delete_from_database()
      return {"message": ITEM_DELETED.format(name)}, 200

    # if doesn't exist, skip deleting 
    return {"message": ITEM_NOT_FOUND}, 404

  # TO ADD OR UPDATE AN ITEM
  @classmethod
  def put(cls, name: str):
    data = Tracker.parser.parse_args()
    # data = request.get_json()
    tracker = TrackerCollection.find_item_by_name(name)

    # if item is not available, add it
    if tracker is None:
      tracker = TrackerCollection(name, **data)
    # if item exists, update it
    else:
      tracker.price = data['price']
      tracker.aimodel_id = data['aimodel_id']
      tracker.position = data['position']
      tracker.datetime_t = data['datetime_t']
    
    # whether item is changed or inserted, it has to be saved to db
    tracker.save_to_database()
    return tracker.json()


class TrackerList(Resource):
  @classmethod
  @jwt_required(optional=True)
  def get(cls):
    user_id = get_jwt_identity()
    trackers = [tracker.json() for tracker in TrackerCollection.find_all()]

    # if user id is given, then display full details
    if user_id:
      return {"trackers": trackers}, 200

    # else display only item name
    return {
      "trackers": [tracker["name"] for tracker in trackers],
      "message": LOGIN_TO_VIEW_DATA
    }, 200