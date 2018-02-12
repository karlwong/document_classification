# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:13:52 2018

@author: karl.wong
"""

import io
import os
import json

service_account_file_path=r'C:\Users\karl.wong\OneDrive\jll_work\Karl_Wong\TA-backup\test\vision\service-account-file-bc1232b7024f.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file_path


def text_detection_jsoncreator(img_path):
    """Detects document features in an image."""
    # Imports the Google Cloud client library

    from google.cloud import vision
    from google.protobuf.json_format import MessageToJson
    client = vision.ImageAnnotatorClient()
    
    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)
    response_json=MessageToJson(response)
    with open(img_path[:img_path.rfind('.')] + '.json', 'w') as outfile:
        json.dump(response_json, outfile)
        outfile.close
    return(response_json)
        
def json_to_html(js,output_fullpath):
    import json2html
    if type(js)==str:
        response_html=json2html.json2html.convert(json = js)
        with open(output_fullpath, 'w', encoding="utf-8") as outfile:
            outfile.write(response_html)
            outfile.close
    else:
        raise "Input should be a string"

    return response_html


def read_vision_text(dic):
    b={}
    if type(dic)==dict:
        i=0
        for page in dic['fullTextAnnotation']['pages']:
            for block in page['blocks']:
                b[i]={}
                j=0
                for paragraph in block['paragraphs']:
                    b[i][j]=''
                    for word in paragraph['words']:
                        w=''
                        for symbol in word['symbols']:
                            try:
                                if len(symbol['property']['detectedBreak']['type'])>0:
                                    w+= symbol['text'] +' '
                                else:
                                    w+=symbol['text']
                            except:
                                w+= symbol['text']
                        b[i][j]+=w
                    j+=1
                i+=1
    return b
        
            
# The name of the image file to annotate
    

#TA_List=[f for f in os.listdir(tar) if f.endswith('.png')]
imgfile_path = os.path.join(
    r'C:\Users\karl.wong\Desktop\Test',
    'Westminster_20170621~712150_TA_page26.png')
response_js=text_detection_jsoncreator(imgfile_path)
response_html=json_to_html(response_js,imgfile_path[:imgfile_path.rfind('.')] + '.html')
response_dic=json.loads(response_js)
response_block_paragraph_text_dic=read_vision_text(response_dic)
