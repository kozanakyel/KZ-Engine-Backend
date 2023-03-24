from flask_restful import Resource, reqparse
from KZ_project.webapi.models.ai_model import AIModelCollection

STORE_ALREADY_EXISTS = "'{}' store already exists!"
ERROR_CREATING_STORE = "An error occurred while creating the store '{}'."
STORE_NOT_FOUND = "'{}' store cannot be found! Please enter a valid store."
STORE_DELETED = "'{}' store has been successfully deleted!"

_store_parser = reqparse.RequestParser()
_store_parser.add_argument(
  "name",
  type=str,
  required=True,
  help= ERROR_CREATING_STORE.format("name")
)

class AIModel(Resource):
  
  @classmethod
  def get(cls):
    name = _store_parser.parse_args()["name"]
    store = AIModelCollection.find_model_by_name(name)
    if store:
      return store.json()
    return {"message" : STORE_NOT_FOUND.format(name)}, 404

  @classmethod
  def post(cls):
    name = _store_parser.parse_args()["name"]

    # find user in database
    print(name)
    if AIModelCollection.find_model_by_name(name):
      return {"message": STORE_ALREADY_EXISTS.format(name)}, 400

    store = AIModelCollection(name)

    try:
      store.save_to_database()
    except:
      return {"message": ERROR_CREATING_STORE.format(name)}, 500
        
    return store.json(), 201


  @classmethod
  def delete(cls, name:str):
    store = AIModelCollection.find_model_by_name(name)
    if store:
      store.delete_from_database()
      return {"message": STORE_DELETED.format(name)}

    return {"message": STORE_NOT_FOUND.format(name)}


class AIModelList(Resource):
  @classmethod
  def get(cls):
    # return {"item": list(map(lambda x: x.json(), ItemModel.query.all()))}
    return {"stores": [store.json() for store in AIModelCollection.find_all()]}
