from flask import Flask, Blueprint
import os


bp = Blueprint('flaskgur', __name__)


from app.flaskgur import flaskgur
