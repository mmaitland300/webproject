import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))
BASE_DIR = "app/flaskgur/"
BASE_DIR2 = "app/faceswap/"

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 25)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS') is not None
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    ADMINS = ['mmaitland300@gmail.com']
    LANGUAGES = ['en', 'es']
    MS_TRANSLATOR_KEY = os.environ.get('MS_TRANSLATOR_KEY')
    ELASTICSEARCH_URL = os.environ.get('ELASTICSEARCH_URL')
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'
    POSTS_PER_PAGE = 25
    SEND_FILE_MAX_AGE_DEFAULT = 1
    AGE_MODEL = basedir + "/age_net.caffemodel"
    AGE_PROTO = basedir + "/age_deploy.prototxt"
    FACE_MODEL = basedir + "/res10_300x300_ssd_iter_140000.caffemodel"
    FACE_PROTO = basedir +  "/deploy.prototxt"
    SHAPE = basedir + "/shape_predictor_68_face_landmarks.dat"
    UPLOAD_DIR = os.path.join(basedir, 'app/pics')
    DATABASE = os.path.join(BASE_DIR, 'flaskgur.db')
    SCHEMA = os.path.join(BASE_DIR, 'schema.sql')
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
    THUMBDIR = os.path.join(basedir, 'thumbnails')
    IMGDIR = os.path.join(basedir, 'images')       
    UPLOAD_FOLDER_VIDEOS = os.path.join(basedir, 'uploaded_videos')
    CARTOONIZED_FOLDER = os.path.join(basedir, 'cartoons')