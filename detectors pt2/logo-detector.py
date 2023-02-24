import requests as requests
# from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
# from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests

# prediction_resource_id = "/subscriptions/f294215e-1528-4d11-a41c-2c5eb70966a0/resourceGroups/USFCapstoneSpring2023/providers/Microsoft.CognitiveServices/accounts/usflogodetector"


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
ENDPOINT = "https://usflogodetector-prediction.cognitiveservices.azure.com/"
prediction_key = "139023d87b174213ab54f4c6db9ff98b"
ProjectID = "2c5d2353-119b-4244-9dd8-754a81b4bae3"
ModelName = "firstpythontest"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    prediction_client = CustomVisionPredictionClient(endpoint=ENDPOINT, credentials=credentials)

    print('Detecting objects in image')

    with open('test.txt') as f:
        image_url = [line.rstrip('\n') for line in f]
    
    
    # image_url = "https://cdn11.bigcommerce.com/s-xlf2wk/images/stencil/1024x1024/products/133/517/USF-United-Start-2-Finish-Green-Shirt-15110__45050.1576079447.jpg?c=2"
    




    image_output_list = []
    image_count = 0
    for image in image_url:
        results = prediction_client.detect_image_url(ProjectID,ModelName,image)


        fig = plt.figure(figsize=(8,8))
        plt.axis('off')



        color='magenta'
        for prediction in results.predictions:
            if (prediction.probability*100) >50:

                url_image = requests.get(image).content
                with open('output.jpg','wb') as handler:
                    handler.write(url_image)
                image_file = 'output.jpg'
                image = Image.open(image_file)
                h, w, ch = np.array(image).shape
                lineWidth = int(w / 100)
                draw = ImageDraw.Draw(image)

                left = prediction.bounding_box.left * w
                top = prediction.bounding_box.top *h
                height = prediction.bounding_box.height*h
                width = prediction.bounding_box.width * w

                points = ((left,top),(left+width,top), (left+width,top+height),(left, top+height) )
                draw.line(points,fill=color, width=lineWidth)
                plt.annotate(prediction.tag_name + ": {0:.2f}%".format(prediction.probability*100), xy = (left,top))
                plt.imshow(image)
                image_count = image_count+1
                outputfile = 'marked_image'+str(image_count)+'.jpg'
                if not os.path.isdir("output-of-logo-detector/"):
                    os.makedirs("output-of-logo-detector/")
                fig.savefig('output-of-logo-detector/'+outputfile)
                print('Results saved in ', 'output-of-logo-detector/'+outputfile)






#     prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
#     predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
#
#     with open(os.path.join(base_image_location, "Test/test_image.jpg"), "rb") as image_contents:
#         results = predictor.classify_image(
#             ProjectID, ModelName, image_contents.read())
#
#         # Display the results.
#         for prediction in results.predictions:
#             print("\t" + prediction.tag_name +
#                   ": {0:.2f}%".format(prediction.probability * 100))
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/