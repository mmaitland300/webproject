import json
import sys
import time
from flask import render_template
from rq import get_current_job
from app import create_app, db
from app.models import User, Post, Task
from app.email import send_email

app = create_app()
app.app_context().push()


def _set_task_progress(progress):
    job = get_current_job()
    if job:
        job.meta['progress'] = progress
        job.save_meta()
        task = Task.query.get(job.get_id())
        task.user.add_notification('task_progress', {'task_id': job.get_id(),
                                                     'progress': progress})
        if progress >= 100:
            task.complete = True
        db.session.commit()


def export_posts(user_id):
    try:
        user = User.query.get(user_id)
        _set_task_progress(0)
        data = []
        i = 0
        total_posts = user.posts.count()
        for post in user.posts.order_by(Post.timestamp.asc()):
            data.append({'body': post.body,
                         'timestamp': post.timestamp.isoformat() + 'Z'})
            time.sleep(5)
            i += 1
            _set_task_progress(100 * i // total_posts)

        send_email('[Microblog] Your blog posts',
                sender=app.config['ADMINS'][0], recipients=[user.email],
                text_body=render_template('email/export_posts.txt', user=user),
                html_body=render_template('email/export_posts.html',
                                          user=user),
                attachments=[('posts.json', 'application/json',
                              json.dumps({'posts': data}, indent=4))],
                sync=True)
    except:
        _set_task_progress(100)
        app.logger.error('Unhandled exception', exc_info=sys.exc_info())

def write_level(text):
    
    letterscount = 0
    wordcount = 1
    sentencecount = 0

    #  lettercount wordcount and sentencecount
    for i in range(len(text)):

        if ((text[i] >= 'a' and text[i] <= 'z') or (text[i] >= 'A' and text[i] <= 'Z')):

            letterscount += 1

        elif (text[i] == ' '):

            wordcount += 1

        elif (text[i] == '.' or text[i] == '!' or text[i] == '?'):

            sentencecount += 1


    #  printf("letters: %i; words: %i; sentences: %i\n", letterscount, wordcount, sentencecount);

    grade = 0.0588 * (100 * float(letterscount) / float(wordcount)) - 0.296 * (100 * float(sentencecount) / float(wordcount)) - 15.8

    if (grade < 16 and grade >= 0):

        return (f"You write at a {int(round(grade))}th grade level")
        
    elif (grade >= 16):
        
        return ("You write at a Post Grad level\n")
        
    else:
        
        return ("Goo Goo Ga Ga\n")



