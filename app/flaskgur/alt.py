import PIL.Image
from PIL import ImageDraw, ImageEnhance
import face_recognition
import cv2
import subprocess
import numpy as np
import os
from flask import current_app

def detect_face(image):
    original_file_path = image
    # Load the jpg file into a numpy array
    image = cv2.imread(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    print("[INFO] recognizing faces...")
    face_locations = face_recognition.face_locations(rgb, model="hog")

    pil_image = PIL.Image.fromarray(rgb)

    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        
        draw.rectangle(((left - 20, top - 20), (right + 20, bottom + 20)), outline=(0, 255, 0), width=20)
    
    pil_image.save(original_file_path)
    
    return "I found {} face(s) in this photograph.".format(len(face_locations))

#detect_face('matt3.jpg')

def detect_age(image):
    original_file_path = image
    # define the list of age buckets our age detector will predict
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
        "(38-43)", "(48-53)", "(60-100)"]

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = current_app.config['FACE_PROTO']
    weightsPath = current_app.config['FACE_MODEL']
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load our serialized age detector model from disk
    print("[INFO] loading age detector model...")
    prototxtPath = current_app.config['AGE_PROTO']
    weightsPath = current_app.config['AGE_MODEL']
    ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the input image and construct an input blob for the image
    image = cv2.imread(image)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > .95:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the ROI of the face and then construct a blob from
            # *only* the face ROI
            face = image[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False)

            # make predictions on the age and find the age bucket with
            # the largest corresponding probability
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            # display the predicted age to our terminal
            text = "{}: {:.2f}%".format(age, ageConfidence * 100)
            print("[INFO] {}".format(text))

            # draw the bounding box of the face along with the associated
            # predicted age
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 3)

    # display the output image
    cv2.imwrite(original_file_path, image)
#detect_age("me.jpg")
import numpy as np
import os, inspect
import math
import matplotlib.pyplot as plt
import scipy
import copy

def edge(image):
    im = cv2.imread(image)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 100)
    edged = PIL.Image.fromarray(edged)
    edged.save(image)

def vintage(image):
    im = cv2.imread(image)
    rows, cols = im.shape[:2]
    # Create a Gaussian filter
    kernel_x = cv2.getGaussianKernel(cols,200)
    kernel_y = cv2.getGaussianKernel(rows,200)
    kernel = kernel_y * kernel_x.T
    filter = 250 * kernel / np.linalg.norm(kernel)
    vintage_im = np.copy(im)
    # for each channel in the input image, we will apply the above filter
    for i in range(3):
        vintage_im[:,:,i] = vintage_im[:,:,i] * filter
    plt.imsave(image, vintage_im)




def gaussianBlur(image):
    initimg = cv2.imread(image)
    x = cv2.GaussianBlur(initimg, (35, 35), 0)
    cv2.imwrite(image, x)

def sepia(image):
    initimg = cv2.imread(image)
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    x = cv2.filter2D(initimg, -1, kernel)
    cv2.imwrite(image, x)

def emboss(image):
    initimg = cv2.imread(image)
    kernel = np.array([[0,-1,-1],
                        [1,0,-1],
                        [1,1,0]])
    x = cv2.filter2D(initimg, -1, kernel)
    cv2.imwrite(image, x)


def brightnessControl(image, level):
    initimg = cv2.imread(image)
    x = cv2.convertScaleAbs(initimg, beta=level)
    cv2.imwrite(image, x)


def sharpen(image):
    initimg = cv2.imread(image)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    x = cv2.filter2D(initimg, -1, kernel)
    cv2.imwrite(image, x)


def enhance(image):
    img = PIL.Image.open(image)
    enh = ImageEnhance.Contrast(img)
    enh.enhance(1.8).save(image)

    
    

from wand.image import Image
from wand.display import display
from wand.color import Color
from wand.drawing import Drawing
import imageio
import click
import numpy as np
import time

def loading_bar(recurse_depth):
    global load_progress
    global start_time
    load_depth=3
    try:
        load_progress
        start_time
    except:
        load_progress = 0
        start_time = time.time()
        print('[' + ' '*(4**load_depth) + ']\r', end='')
    if recurse_depth <= load_depth:
        load_progress += 4**(load_depth - recurse_depth)
        cur_time = time.time()
        time_left = 4**load_depth*(cur_time - start_time)/load_progress \
                  - cur_time + start_time
        print('[' + '='*load_progress \
                  + ' '*(4**load_depth - load_progress) \
                  + '] ' \
                  + 'time left: {} secs'.format(int(time_left)).ljust(19) \
                  + '\r', end='')

class QuadArt:
    def __init__(self, std_thresh=10, draw_type='circle', max_recurse=None):
        self.img = None
        self.canvas = None
        self.draw = None
        self.std_thresh = std_thresh
        self.draw_type = draw_type
        self.recurse_depth = 0
        self.max_recurse_depth = max_recurse

    def recursive_draw(self, x, y, w, h):
        '''Draw the QuadArt recursively
        '''

        if (self.max_recurse_depth == 0 or self.recurse_depth < self.max_recurse_depth) \
        and self.too_many_colors(int(x), int(y), int(w), int(h)):
            self.recurse_depth += 1

            self.recursive_draw(x,         y,         w/2.0, h/2.0)
            self.recursive_draw(x + w/2.0, y,         w/2.0, h/2.0)
            self.recursive_draw(x,         y + h/2.0, w/2.0, h/2.0)
            self.recursive_draw(x + w/2.0, y + h/2.0, w/2.0, h/2.0)

            self.recurse_depth -= 1

            if self.recurse_depth == 3:
                loading_bar(self.recurse_depth)
        else:
            self.draw_avg(x, y, w, h)

            if self.recurse_depth < 3:
                loading_bar(self.recurse_depth)

    def too_many_colors(self, x, y, w, h):
        if w * self.output_scale <= 2 or w <= 2:
            return False
        img = self.img[y:y+h,x:x+w]
        red = img[:,:,0]
        green = img[:,:,1]
        blue = img[:,:,2]

        red_avg = np.average(red)
        green_avg = np.average(green)
        blue_avg = np.average(blue)

        if red_avg >= 254 and green_avg >= 254 and blue_avg >= 254:
            return False

        if 255 - red_avg < self.std_thresh and 255 - green_avg < self.std_thresh \
                                           and 255 - blue_avg < self.std_thresh:
            return True

        red_std = np.std(red)
        if red_std > self.std_thresh:
            return True

        green_std = np.std(green)
        if green_std > self.std_thresh:
            return True

        blue_std = np.std(blue)
        if blue_std > self.std_thresh:
            return True

        return False

    def draw_avg(self, x, y, w, h):
        avg_color = self.get_color(int(x), int(y), int(w), int(h))
        self.draw_in_box(avg_color, x, y, w, h)
        return avg_color

    def get_color(self, x, y, w, h):
        img = self.img[y : y + h,
                       x : x + w]
        red = np.average(img[:,:,0])
        green = np.average(img[:,:,1])
        blue = np.average(img[:,:,2])
        color = Color('rgb(%s,%s,%s)' % (red, green, blue))
        return color

    def draw_in_box(self, color, x, y, w, h):
        if self.draw_type == 'circle':
            self.draw_circle_in_box(color, x, y, w, h)
        else:
            self.draw_square_in_box(color, x, y, w, h)

    def draw_circle_in_box(self, color, x, y, w, h):
        x *= self.output_scale
        y *= self.output_scale
        w *= self.output_scale
        h *= self.output_scale

        self.draw.fill_color = color
        self.draw.circle((int(x + w/2.0), int(y + h/2.0)),
                         (int(x + w/2.0), int(y)))

    def draw_square_in_box(self, color, x, y, w, h):
        x *= self.output_scale
        y *= self.output_scale
        w *= self.output_scale
        h *= self.output_scale

        self.draw.fill_color = color
        self.draw.rectangle(x, y, x + w, y + h)

    def width(self):
        return self.img.shape[1]

    def scale_width(self):
        return self.width() * self.output_scale

    def height(self):
        return self.img.shape[0]

    def scale_height(self):
        return self.height() * self.output_scale

    def generate(self, filename,
                 left=None, right=None, up=None, down=None,
                 output_size=512):
        self.img = imageio.imread(filename)
        left  = 0             if left  is None else int(self.width()  * float(left))
        right = self.width()  if right is None else int(self.width()  * float(right))
        up    = 0             if up    is None else int(self.height() * float(up))
        down  = self.height() if down  is None else int(self.height() * float(down))
        self.img = self.img[up:down,left:right]
        if self.width() < self.height():
            difference = self.height() - self.width()
            subtract_top = int(difference/2)
            subtract_bot = difference - subtract_top
            self.img = self.img[subtract_top:-subtract_bot,:]
        elif self.height() < self.width():
            difference = self.width() - self.height()
            subtract_left = int(difference/2)
            subtract_right = difference - subtract_left
            self.img = self.img[:,subtract_left:-subtract_right]

        self.output_scale = float(output_size) / self.width()

        self.canvas = Image(width = output_size,
                            height = output_size,
                            background = Color('white'))
        self.canvas.format = 'png'
        self.draw = Drawing()
        self.recursive_draw(0, 0, self.width(), self.height())
        self.draw(self.canvas)

    def display(self):
        display(self.canvas)

    def save(self, filename):
        self.canvas.save(filename=filename)



def mainquad(filename, left=None, right=None, up=None, down=None, size=512, draw_type='circle', thresh=10, max_recurse=0):
    quadart = QuadArt(std_thresh=thresh, draw_type=draw_type, max_recurse=max_recurse)
    quadart.generate(filename, left=left, right=right,
                               up=up, down=down,
                               output_size=size)
    
    quadart.save(filename)



# class PyGram:
	
# 	def __init__(self, filename):
# 		self.filename = filename
# 		self.im = False
		
# 	def image(self):
# 		if not self.im:
# 			self.im = PIL.Image.open(self.filename)
# 		return self.im
	
# 	def execute(self, command, **kwargs):
# 		default = dict(
# 			filename=self.filename,
# 			width = self.image().size[0],
# 			height = self.image().size[1]
# 		)
# 		format = dict(default.items() | kwargs.items())
# 		command = command.format(**format)
# 		error = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
# 		return error
		
# 	def colortone(self, color, level, type = 0):
		
# 		arg0 = level
# 		arg1 = 100 - level
# 		if type == 0:
# 			negate = '-negate'
# 		else:
# 			negate = ''

# 		self.execute("convert {filename} \( -clone 0 -fill '{color}' -colorize 100% \) \( -clone 0 -colorspace gray {negate} \) -compose blend -define compose:args={arg0},{arg1} -composite {filename}",
# 			color = color,
# 			negate = negate,
# 			arg0 = arg0,
# 			arg1 = arg1
# 		)

# # Decorations

# class Vignette(PyGram):
#     def vignette(self, color_1='none', color_2='black', crop_factor=1.5):
#         crop_x = math.floor(self.image().size[0] * crop_factor)
#         crop_y = math.floor(self.image().size[1] * crop_factor)

#         self.execute("convert \( {filename} \) \( -size {crop_x}x{crop_y} radial-gradient:{color_1}-{color_2} -gravity center -crop {width}x{height}+0+0 +repage \) -compose multiply -flatten {filename}",
# 			crop_x = crop_x,
# 			crop_y = crop_y,
# 			color_1 = color_1,
# 			color_2 = color_2,
# 		)



# class Frame(PyGram):
#     def frame(self, frame):
#         path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#         self.execute(
#             "convert {filename} \( '{frame}' -resize {width}x{width}! -unsharp 1.5x1.0+1.5+0.02 \) -flatten {filename}",
#             frame=os.path.join(path, "frames", frame)
#         )

# class Border(PyGram):
#     def border(self, color='black', width=20):
#         self.execute("convert {filename} -bordercolor {color} -border {bwidth}x{bwidth} {filename}",
#                      color=color,
#                      bwidth=width
#         )


# # Filters

# class Toaster(Vignette, Border):
#     def apply(self):
#         self.colortone('#330000', 50, 0)
#         self.execute("convert {filename} -modulate 150,80,100 -gamma 1.2 -contrast -contrast {filename}")
#         self.vignette('none', 'LavenderBlush3')
#         self.vignette('#ff9966', 'none')
#         self.border('white')

# class Lomo(Vignette):

# 	def apply(self):
# 		self.execute("convert {filename} -channel R -level 33% -channel G -level 33% {filename}")
# 		self.vignette()

# class Kelvin(Border):

# 	def apply(self):
# 		self.execute(
#             "convert \( {filename} -auto-gamma -modulate 120,50,100 \) \( -size {width}x{height} -fill 'rgba(255,153,0,0.5)' -draw 'rectangle 0,0 {width},{height}' \) -compose multiply {filename}")
# 		self.border()

# class Gotham(Border):
#     def apply(self):
#         self.execute(
#             "convert {filename} -modulate 120,10,100 -fill '#222b6d' -colorize 20 -gamma 0.5 -contrast -contrast {filename}")
#         self.border()

# class Nashville(Border):

# 	def apply(self):
# 		self.colortone('#222b6d', 50, 0)
# 		self.colortone('#f7daae', 120, 1)
# 		self.execute("convert {filename} -contrast -modulate 100,150,100 -auto-gamma {filename}")
# 		self.border()
# # x = Lomo("sharp.jpg")
# # x.apply()