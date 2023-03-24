from models.user import UserModel
#compares strs and encodings -> returns true if strs/encodings are same
#from werkzeug.security import safe_str_cmp
import hmac

def safe_str_cmp(a, b):
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0

 
def authenticate(username, password):
    #user -> entity
    user = UserModel.find_by_username(username)
    if user and safe_str_cmp(user.password, password):
        return user

def identity(payload):
    user_id = payload['identity']
    return UserModel.find_by_id(user_id)

    
