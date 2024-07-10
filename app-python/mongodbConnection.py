import pymongo # type: ignore

url = 'mongodb://localhost:27017'
client= pymongo.MongoClient(url)
db =client['yazlab']