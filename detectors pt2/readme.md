# To install before running detector scripts:

-pip install azure-cognitiveservices-vision-customvision
-pip install --upgrade azure-cognitiveservices-vision-computervision
-pip install pillow
-python -m pip install -U pip
-python -m pip install -U matplotlib



#IGNORE FOR NOW
What logo-detector does:
1) Accesses "test.txt" file and fetches urls line by line
2) each image is analyzed and saved separately.
3) If not present, a directory "/output-of-logo-detector" is created with all the marked images are saved in it.

What text-detector does:
1) Accesses "test.txt" file and fetches urls line by line
2) Text in each image is saved in a separate ".txt" file
3) Text is saved by order written
4) If not present, a directory "/output-of-text-detector" is created with each text file saved in it.
