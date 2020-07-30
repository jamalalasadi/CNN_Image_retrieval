import os 
import flask
import werkzeug
import time
import pathlib

app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    files_ids = list(flask.request.files)
    print("\nNumber of Received Images : ", len(files_ids))
    image_num = 1
    for file_id in files_ids:
        print("\nSaving Image ", str(image_num), "/", len(files_ids))
        imagefile = flask.request.files[file_id]
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("Image Filename : " + imagefile.filename)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imagefile.save(timestr+'_'+filename)
        image_num = image_num + 1
    print("\n")   
    return str((os.path.join(str(pathlib.Path().absolute()), timestr+'_'+filename)))
    

app.run(host='192.168.1.48', port=33, debug=True)
