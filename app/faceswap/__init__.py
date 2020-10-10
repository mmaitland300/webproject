from flask import Flask, Blueprint


bp = Blueprint('faceswap', __name__)


from app.faceswap import faceswap
