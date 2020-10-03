""" Main view definitions.
"""
import time
import os
import sqlite3
from hashlib import md5
from PIL import Image
from flask_login import current_user, login_required
from app.main.forms import EditProfileForm, EmptyForm, PostForm, SearchForm, \
    MessageForm, EditJournalForm
from app.models import User, Post, Message, Notification

from flask import request, g, redirect, url_for, abort, render_template, send_from_directory, current_app
from werkzeug.utils import secure_filename
from app.flaskgur.alt import detect_face, detect_age, mainquad, edge, vintage, sepia, gaussianBlur, emboss, sharpen, enhance
from app.flaskgur import bp

def check_extension(extension):
    """
    Make sure extension is in the ALLOWED_EXTENSIONS set
    """
    return extension in current_app.config['ALLOWED_EXTENSIONS']

def connect_db():
    """ Connect to the SQLite database.
    """
    query = open(current_app.config['SCHEMA'], 'r').read()
    conn = sqlite3.connect(current_app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.executescript(query)
    conn.commit()
    cursor.close()
    return sqlite3.connect(current_app.config['DATABASE'])

def get_last_pics():
    """ Return a list of the last 25 uploaded images
    """
    cur = g.db.execute('select filename from pics order by created_on desc limit 25')
    filenames = [row[0] for row in cur.fetchall()]
    return filenames


def add_pic(filename):
    """ Insert filename into database
    """
    g.db.execute('insert into pics (filename) values (?)', [filename])
    g.db.commit()

def remove_pic(filename):
    g.db.execute('DELETE FROM pics WHERE filename = "(filename)"', [filename])
    g.db.commit()

def gen_thumbnail(filename):
    """ Generate thumbnail image
    """
    height = width = 200
    original = Image.open(os.path.join(current_app.config['UPLOAD_DIR'], filename))
    thumbnail = original.resize((width, height), Image.ANTIALIAS)
    thumbnail.save(os.path.join(current_app.config['UPLOAD_DIR'], 'thumb_'+filename))


@bp.before_request
def before_request():
    """ Executes before each request.
    Taken from flask example app
    """
    g.db = connect_db()

@bp.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must- \
    revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@bp.teardown_request
def teardown_request(err):
    """ Executes after each request, regardless of whether
    there was an exception or not.
    """
    database = getattr(g, 'db', None)
    if database is not None:
        database.close()

@bp.errorhandler(404)
def page_not_found(err):
    """ Redirect to 404 on error.
    """
    return render_template('flaskgur/404.html'), 404

@bp.route('/matgur', methods=['GET', 'POST'])
def upload_pic():
    """ Default route.
    """
    if request.method == 'POST':
        image_file = request.files['file']
        try:
            extension = image_file.filename.rsplit('.', 1)[1].lower()
        except IndexError:
            abort(404)
        if image_file and check_extension(extension):
            # Salt and hash the file contents
            filename = secure_filename(image_file.filename)
            image_file.seek(0) # Move cursor back to beginning so we can write to disk
            image_file.save(os.path.join(current_app.config['UPLOAD_DIR'], filename))
            try:
                add_pic(filename)
            except sqlite3.IntegrityError:
                return redirect(url_for('flaskgur.show_pic', filename=filename))
            gen_thumbnail(filename)
            return redirect(url_for('flaskgur.show_pic', filename=filename))
        else: # Bad file extension
            abort(404)
    else:
        return render_template('flaskgur/upload.html', pics=get_last_pics())

@bp.route('/show')
def show_pic():
    """ Show a file specified by GET parameter.
    """
    filename = request.args.get('filename', '')
    return render_template('flaskgur/upload.html', filename=filename)

@bp.route('/pics/<filename>')
def return_pic(filename):
    """ Show just the image specified.
    """
    return send_from_directory(current_app.config['UPLOAD_DIR'], secure_filename(filename))


    


@bp.route('/detect/<filename>')
def face(filename):

    detect_face(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
    
    
    gen_thumbnail(filename)
    
    return redirect(url_for('flaskgur.show_pic', filename=filename))


@bp.route('/age/<filename>')
def age(filename):

    detect_age(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
    
    gen_thumbnail(filename)
    
    return redirect(url_for('flaskgur.show_pic', filename=filename))


@bp.route('/quad/<filename>')
def quadart(filename):

    mainquad(filename=(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename))))
    
    gen_thumbnail(filename)
    
    return redirect(url_for('flaskgur.show_pic', filename=filename))

@bp.route('/edge/<filename>')
def lined(filename):
    
    edge(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
    
    gen_thumbnail(filename)

    return redirect(url_for('flaskgur.show_pic', filename=filename))


@bp.route('/vint/<filename>')
def vint(filename):
    
    vintage(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
    
    gen_thumbnail(filename)

    return redirect(url_for('flaskgur.show_pic', filename=filename))


@bp.route('/sep/<filename>')
def sep(filename):
    
    sepia(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
    
    gen_thumbnail(filename)

    return redirect(url_for('flaskgur.show_pic', filename=filename))

@bp.route('/embo/<filename>')
def embo(filename):
    
    emboss(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
    
    gen_thumbnail(filename)

    return redirect(url_for('flaskgur.show_pic', filename=filename))

@bp.route('/blur/<filename>')
def blur(filename):
    
    gaussianBlur(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
    
    gen_thumbnail(filename)

    return redirect(url_for('flaskgur.show_pic', filename=filename))

@bp.route('/sharp/<filename>')
def sharp(filename):
    
    sharpen(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
    
    gen_thumbnail(filename)

    return redirect(url_for('flaskgur.show_pic', filename=filename))

@bp.route('/enhance/<filename>')
def enhan(filename):
    
    enhance(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
    
    gen_thumbnail(filename)

    return redirect(url_for('flaskgur.show_pic', filename=filename))





@bp.route('/del/<filename>')
def remove(filename):
    os.remove(os.path.join(current_app.config['UPLOAD_DIR'], filename))
    os.remove(os.path.join(current_app.config['UPLOAD_DIR'], "thumb_" + filename))
    g.db.execute(f'DELETE FROM pics WHERE filename = "{filename}"')
    g.db.commit()
    return redirect(url_for('flaskgur.upload_pic'))





# @bp.route('/goth/<filename>')
# def goth(filename):
    
    
#     beaut = Gotham(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
#     beaut.apply()
    
#     gen_thumbnail(filename)

#     return redirect(url_for('flaskgur.show_pic', filename=filename))

# @bp.route('/nash/<filename>')
# def nash(filename):


#     beaut = Nashville(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
#     beaut.apply()

#     gen_thumbnail(filename)

#     return redirect(url_for('flaskgur.show_pic', filename=filename))

# @bp.route('/toast/<filename>')
# def toast(filename):


#     beaut = Toaster(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
#     beaut.apply()

#     gen_thumbnail(filename)

#     return redirect(url_for('flaskgur.show_pic', filename=filename))

# @bp.route('/lomo/<filename>')
# def lomo(filename):


#     beaut = Lomo(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
#     beaut.apply()

#     gen_thumbnail(filename)

#     return redirect(url_for('flaskgur.show_pic', filename=filename))

# @bp.route('/kelvin/<filename>')
# def kelvin(filename):


#     beaut = Kelvin(os.path.join(current_app.config['UPLOAD_DIR'], secure_filename(filename)))
#     beaut.apply()

#     gen_thumbnail(filename)

#     return redirect(url_for('flaskgur.show_pic', filename=filename))