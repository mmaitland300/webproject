from app import create_app, db, csrf
from app.models import User, Post, Message, Notification, Task

app = create_app()