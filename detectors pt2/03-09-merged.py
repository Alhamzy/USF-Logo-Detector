# custom vision dependancies
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os, time, uuid, requests
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# computer vision dependancies
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from msrest.authentication import ApiKeyCredentials
from array import array
import re

# custom vision credentials
customvision_endpoint = "https://usflogodetector-prediction.cognitiveservices.azure.com/"
prediction_key = "139023d87b174213ab54f4c6db9ff98b"
ProjectID = "2c5d2353-119b-4244-9dd8-754a81b4bae3"
ModelName = "firstpythontest"

# computer vision credentials
subscription_key  = "f990511341b449048664f7a18991401c"
computervision_endpoint = "https://usf-textdetector.cognitiveservices.azure.com/"



# start detection
if __name__ == '__main__':

    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    prediction_client = CustomVisionPredictionClient(endpoint=customvision_endpoint, credentials=credentials)
    input_file = 'urls.txt'
    copyrighted = 'copyrighted.txt'
    print('Detecting objects from ',input_file,':')
    
    
    with open(input_file) as f:
        image_url_list = [line.rstrip('\n') for line in f]


    image_count = 0
    for image_url in image_url_list:
        # preparing custom vision client
        results = prediction_client.detect_image_url(ProjectID,ModelName,image_url) # get prediction response for image url detection
        
        # preparing computer vision client
        computervision_client = ComputerVisionClient(computervision_endpoint, CognitiveServicesCredentials(subscription_key))
        read_response=computervision_client.read(url=image_url,raw=True) # get response from "read" call, for text detection

        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]
        # Call the "GET" API and wait for it to retrieve the results 
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)
        
        
        #prepare output properties
        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        color='magenta'
        
        for prediction in results.predictions:
            if (prediction.probability*100) >50: # logo detected
                # start logo marking
                url_image = requests.get(image_url).content
                with open('output.jpg','wb') as handler:
                    handler.write(url_image)
                image_file = 'output.jpg'
                image_handler = Image.open(image_file)
                dimensions = np.array(image_handler).shape # dimensions => (height,width,channel)
                h = dimensions[0]
                w = dimensions[1]
                if len(dimensions) == 3:
                    ch = dimensions[2]
                lineWidth = int(w / 200)
                draw = ImageDraw.Draw(image_handler)
                
                left = prediction.bounding_box.left * w
                top = prediction.bounding_box.top *h
                height = prediction.bounding_box.height*h
                width = prediction.bounding_box.width * w
                
                points = ((left,top),(left+width,top), (left+width,top+height),(left, top+height),(left,top),(left+width,top))

                draw.line(points,fill=color, width=lineWidth)
                plt.annotate(prediction.tag_name + ": {0:.2f}%".format(prediction.probability*100), xy = (left,top))
                plt.imshow(image_handler)
                image_count = image_count+1
                
                #check for existance of sensitive words next to logo
                if read_result.status == OperationStatusCodes.succeeded:
                    # check if any text was detected
                    text_result = read_result.analyze_result.read_results[0]
                    if len(text_result.lines) == 0:
                        print("No text on image.")
                        outputfile = 'marked_image'+str(image_count)+'.jpg'
                        if not os.path.isdir("output-of-logo-detector/"):
                            os.makedirs("output-of-logo-detector/")
                        fig.savefig('output-of-logo-detector/'+outputfile)
                        print('Results saved in ', 'output-of-logo-detector/'+outputfile)
                        continue
                
                    with open(copyrighted) as file:
                        copyrighted_phrases = [line.rstrip('\n') for line in file]
                        
                    for text_result in read_result.analyze_result.read_results: # for each line of text, print output and mark on image
                        for line in text_result.lines: 
                            # check each detected line of text in image for copyrighted text
                            for phrase in copyrighted_phrases:
                                if(phrase == ''):
                                    continue
                                if re.search(phrase, line.text, re.IGNORECASE) is not None: # if sensitive word found
                                    points = (line.bounding_box[0],line.bounding_box[1],line.bounding_box[2],line.bounding_box[3],line.bounding_box[4],line.bounding_box[5],line.bounding_box[6],line.bounding_box[7],line.bounding_box[0],line.bounding_box[1]) 
                                    draw.line(points,fill=color, width=lineWidth)
                                    plt.annotate(line.text, xy = (line.bounding_box[0],line.bounding_box[1]))
                                    plt.imshow(image_handler)
                # save marked image
                outputfile = 'marked_image'+str(image_count)+'.jpg'
                if not os.path.isdir("output-of-final-detector/"):
                    os.makedirs("output-of-final-detector/")
                fig.savefig('output-of-final-detector/'+outputfile)
                print('Results saved in ', 'output-of-detector/'+outputfile)
