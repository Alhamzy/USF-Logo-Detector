from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
from PIL import Image
import sys
import time

subscription_key  = "f990511341b449048664f7a18991401c"
endpoint = "https://usf-textdetector.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# OCR: Read File using the Read API, extract text - remote
# This example will extract text in an image, then print results, line by line.
# This API call can also extract handwriting style text (not shown).

print("===== Read File - remote =====")
# Get an image with text
with open('test.txt') as f:
        input_url_list = [line.rstrip('\n') for line in f]
# read_image_url = 'https://media.cnn.com/api/v1/images/stellar/prod/160122124623-01-national-handwriting-day.jpg?q=w_3264,h_1836,x_0,y_0,c_fill'

image_output_list = []
image_count = 0

for image_url in input_url_list:
    read_response=computervision_client.read(url=image_url,raw=True)


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

    # Print the detected text, line by line
    image_count=image_count+1
    print("\nPrinting text for image number "+str(image_count))
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                print(line.text)
                print(line.bounding_box)
    print()
    
    # save fetched text to separate file for each image
    if not os.path.isdir("output-of-text-detector/"):
                    os.makedirs("output-of-text-detector/")
    with open('output-of-text-detector/text_in_image'+str(image_count), 'w') as f:
        for line in text_result.lines:
            f.write(line.text+"\n")

    print("Done analyzing image number "+str(image_count))
