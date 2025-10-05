import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')

@app.route('/userlog.html')
def demo():
    return render_template('userlog.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])

        # apply the sharpening kernel to the image
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)



        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'DiseaseDetection-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 6, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        diseasename=" "
        rem=" "
        rem1=" "
        str_label=" "
        accuracy=""
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Healthy'
            elif np.argmax(model_out) == 1:
                str_label = 'Bacterial'
            elif np.argmax(model_out) == 2:
                str_label = 'curl virus'
            elif np.argmax(model_out) == 3:
                str_label = 'Spectoria'
            elif np.argmax(model_out) == 4:
                str_label = 'Leafmold'
            elif np.argmax(model_out) == 5:
                str_label = 'mosaic_virus'

            if str_label == 'Bacterial':
                diseasename = "Bacterial Spot "
                print("The predicted image of the Bacterial is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the Bacterial is with a accuracy of {}%".format(model_out[1]*100)
                rem = "The remedies for Bacterial Spot are:\n\n "
                rem1 = [" Discard or destroy any affected plants",  
                "Do not compost them.", 
                "Rotate yoour tomato plants yearly to prevent re-infection next year.", 
                "Use copper fungicites"]
               
            elif str_label == 'curl virus':
                diseasename = "Yellow leaf curl virus "
                print("The predicted image of the curl virus is with a accuracy of {} %".format(model_out[2]*100))
                accuracy="The predicted image of the curl virus is with a accuracy of {}%".format(model_out[2]*100)
                rem = "The remedies for Yellow leaf curl virus are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates", 
                "carbametes during the seedliing stage.", "Use copper fungicites"]
                
            elif str_label == 'Spectoria':
                diseasename = "Spectoria "
                print("The predicted image of the Spectoria is with a accuracy of {} %".format(model_out[3]*100))
                accuracy="The predicted image of the Spectoria is with a accuracy of {}%".format(model_out[3]*100)
                rem = "The remedies for spectoria are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates",
                "carbametes during the seedliing stage.",
                "Use copper fungicites"]
                
                
            elif str_label == 'Healthy':
                status= 'Healthy'
                print("The predicted image of the Healthy is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the Healthy is with a accuracy of {}%".format(model_out[0]*100)
                
                
            elif str_label == 'Leafmold':
                diseasename = "Leafmold"
                print("The predicted image of the Leafmold is with a accuracy of {} %".format(model_out[4]*100))
                accuracy="The predicted image of the Leafmold is with a accuracy of {}%".format(model_out[4]*100)
                rem = "The remedies for Leafmold are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]
                
                
            elif str_label == 'mosaic_virus':
                diseasename = "mosaic_virus"
                print("The predicted image of the mosaic_virus is with a accuracy of {} %".format(model_out[5]*100))
                accuracy="The predicted image of the mosaic_virus is with a accuracy of {}%".format(model_out[5]*100)
                rem = "The remedies for  mosaic_virus are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates",
                "carbametes during the seedliing stage.",
                "Use copper fungicites"]
                
            

        return render_template('results.html', status=str_label,accuracy=accuracy, disease=diseasename, remedie=rem, remedie1=rem1, ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
       
    return render_template('index.html')

@app.route('/live')
def live():
    dirPath = "static/images"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)

    vs = cv2.VideoCapture(0)
    while True:
        ret, image = vs.read()
        if not ret:
            break
        cv2.imshow('Leaf Disease', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('result.png', image)
            break
    vs.release()
    cv2.destroyAllWindows()
    
    dst = "static/images"

    shutil.copy('result.png', dst)
    
    verify_dir = 'static/images'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'DiseaseDetection-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
    def process_verify_data():
        verifying_data = []
        for img in os.listdir(verify_dir):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    
    tf.compat.v1.reset_default_graph()
    #tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 6, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')


    fig = plt.figure()
    diseasename=" "
    rem=" "
    rem1=" "
    str_label=" "
    accuracy=""
    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))

        if np.argmax(model_out) == 0:
            str_label = 'Healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'Bacterial'
        elif np.argmax(model_out) == 2:
            str_label = 'curl virus'
        elif np.argmax(model_out) == 3:
            str_label = 'Spectoria'
        elif np.argmax(model_out) == 4:
            str_label = 'Leafmold'
        elif np.argmax(model_out) == 5:
            str_label = 'mosaic_virus'

        if str_label == 'Bacterial':
            diseasename = "Bacterial Spot "
            print("The predicted image of the Bacterial is with a accuracy of {} %".format(model_out[1]*100))
            accuracy="The predicted image of the Bacterial is with a accuracy of {}%".format(model_out[1]*100)
            rem = "The remedies for Bacterial Spot are:\n\n "
            rem1 = [" Discard or destroy any affected plants",  
            "Do not compost them.", 
            "Rotate yoour tomato plants yearly to prevent re-infection next year.", 
            "Use copper fungicites"]
            
        elif str_label == 'curl virus':
            diseasename = "Yellow leaf curl virus "
            print("The predicted image of the curl virus is with a accuracy of {} %".format(model_out[2]*100))
            accuracy="The predicted image of the curl virus is with a accuracy of {}%".format(model_out[2]*100)
            rem = "The remedies for Yellow leaf curl virus are: "
            rem1 = [" Monitor the field, handpick diseased plants and bury them.",
            "Use sticky yellow plastic traps.", 
            "Spray insecticides such as organophosphates", 
            "carbametes during the seedliing stage.", "Use copper fungicites"]
            
        elif str_label == 'Spectoria':
            diseasename = "Spectoria "
            print("The predicted image of the Spectoria is with a accuracy of {} %".format(model_out[3]*100))
            accuracy="The predicted image of the Spectoria is with a accuracy of {}%".format(model_out[3]*100)
            rem = "The remedies for spectoria are: "
            rem1 = [" Monitor the field, handpick diseased plants and bury them.",
            "Use sticky yellow plastic traps.", 
            "Spray insecticides such as organophosphates",
            "carbametes during the seedliing stage.",
            "Use copper fungicites"]
            
            
        elif str_label == 'Healthy':
            status= 'Healthy'
            print("The predicted image of the Healthy is with a accuracy of {} %".format(model_out[0]*100))
            accuracy="The predicted image of the Healthy is with a accuracy of {}%".format(model_out[0]*100)
            
            
        elif str_label == 'Leafmold':
            diseasename = "Leafmold"
            print("The predicted image of the Leafmold is with a accuracy of {} %".format(model_out[4]*100))
            accuracy="The predicted image of the Leafmold is with a accuracy of {}%".format(model_out[4]*100)
            rem = "The remedies for Leafmold are: "
            rem1 = [" Monitor the field, remove and destroy infected leaves.",
            "Treat organically with copper spray.",
            "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]
            
            
        elif str_label == 'mosaic_virus':
            diseasename = "mosaic_virus"
            print("The predicted image of the mosaic_virus is with a accuracy of {} %".format(model_out[5]*100))
            accuracy="The predicted image of the mosaic_virus is with a accuracy of {}%".format(model_out[5]*100)
            rem = "The remedies for  mosaic_virus are: "
            rem1 = [" Monitor the field, handpick diseased plants and bury them.",
            "Use sticky yellow plastic traps.", 
            "Spray insecticides such as organophosphates",
            "carbametes during the seedliing stage.",
            "Use copper fungicites"]
                
            
    return render_template('results.html', status1=str_label,accuracy=accuracy, disease1=diseasename, remedie1=rem, remedie11=rem1, ImageDisplay5="http://127.0.0.1:5000/static/images/result.png")


@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
