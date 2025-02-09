from os import environ

# constants

KOJI_HUB_URL = "https://koji.fedoraproject.org/kojihub"


# config of model/server

HOST = environ.get("HOST", "localhost")
PORT = int(environ.get("PORT", 44882))
