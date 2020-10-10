import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, current_app, Blueprint
from flask import Blueprint, current_app
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from PIL import Image
from app.faceswap.swap import swap
from app.faceswap import bp






@bp.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must- \
    revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']
	
@bp.route('/faceload')
def upload_form():
	return render_template('faceswap/upload.html')

@bp.route('/faceupload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if len(os.listdir(current_app.config['IMGDIR'])) < 2:
            for upload in request.files.getlist('images'):
                filename = upload.filename
                # Always a good idea to secure a filename before storing it
                filename = secure_filename(filename)
                # This is to verify files are supported
                ext = os.path.splitext(filename)[1][1:].strip().lower()
                if ext in set(['jpg', 'jpeg', 'png']):
                    print('File supported moving on...')
                else:
                    return render_template('error.html', message='Uploaded files are not supported...')
                destination = '/'.join([current_app.config['IMGDIR'], filename])
                # Save original image
                upload.save(destination)
                # Save a copy of the thumbnail image
                image = Image.open(destination)
                image.thumbnail((300, 170))
                image.save('/'.join([current_app.config['THUMBDIR'], filename]))
            return redirect(url_for('faceswap.upload'))
        else:
            return render_template("faceswap/upload.html", warning="Only upload two images!")
    return render_template('faceswap/upload.html')

@bp.route('/fgallery')
def gallery():
    thumbnail_names = os.listdir(current_app.config['THUMBDIR'])
    return render_template('faceswap/gallery.html', thumbnail_names=thumbnail_names)

@bp.route('/fthumbnails/<filename>')
def thumbnails(filename):
    return send_from_directory(current_app.config['THUMBDIR'], filename)

@bp.route('/fimages/<filename>')
def images(filename):
    return send_from_directory(current_app.config['IMGDIR'], filename)



@bp.route('/fdisplay/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('faceswap.images', filename=filename), code=301)

@bp.route('/fswap')
def swapface():
    filename = os.listdir(current_app.config['IMGDIR'])
    filename1 = current_app.config['IMGDIR'] + '/' + filename[0]
    filename2 = current_app.config['IMGDIR'] + '/' + filename[1]
    try:
        swap(filename1, filename2)
        image = Image.open(current_app.config['IMGDIR'] + '/result.jpg')
        image.thumbnail((300, 170))
        image.save('/'.join([current_app.config['THUMBDIR'], '/result.jpg']))
        return redirect(url_for('faceswap.display_image', filename='result.jpg'), code=301)
    except ValueError:
        return render_template("faceswap/gallery.html", warning="No face found please upload a clearer photo")

@bp.route('/fnew')
def newswap():
    images = os.listdir(current_app.config['IMGDIR'])
    for i in images:
        os.remove(os.path.join(current_app.config['IMGDIR'], i))
    thumbs = os.listdir(current_app.config['THUMBDIR'])
    for t in thumbs:
        os.remove(os.path.join(current_app.config['THUMBDIR'], t))
    return render_template('faceswap/upload.html')


