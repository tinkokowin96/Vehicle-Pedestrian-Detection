#To_Do : TO Consifer for Valadation Dataset

import os
import json
import xml.etree.ElementTree as ET
import torch

kitti_label = {'car', 'van', 'truck','pedestrian', 'person_sitting', 'cyclist', 'tram','misc','dontcare'}
label_map = {v:i+1 for i,v in enumerate (kitti_label)}
label_map['background'] = 0

def txt_to_xml (sourcedir,outputdir):
    file_list = []
    dir_list  = []
    for root,dir,file in os.walk(sourcedir):
        for name in file:
            file_list.append(name)
            dir_list.append(os.path.join(root,name))
    

    num_file = len(file_list)
    for i in range(num_file):
        file_name = file_list[i].replace('.txt','')
        
        xml_file  = open(os.path.join(outputdir,file_name+'.xml'),'w')
        
        xml_file.write('<?xml version="1.0" encoding="UTF-8"?>')
        xml_file.write('<annotation>')
        with open(os.path.join(dir_list[i]),'r') as j:
            obj = j.read().splitlines()
            
        obj_label = list()
        truncate  = list()
        occlusion = list()
        bndbox    = list() 
        index = 0
        for _ in obj:
            all_spec = [i for i in obj[index].split()]

            obj_label.append(all_spec[0])
            truncate.append(all_spec[1])
            occlusion.append(all_spec[2])
            bndbox.append(all_spec[4:8])
            
            index += 1
        for i in range(index):       
            xml_file.write('  '+'<object>')
            xml_file.write('   '+'<name>'+obj_label[i]+'</name>')
            xml_file.write('   '+'<truncate>'+truncate[i]+'</truncate>')
            xml_file.write('   '+'<occlusion>'+occlusion[i]+'</occlusion>')
            xml_file.write('   '+'<bndbox>'+'\n')
            xml_file.write('    '+'<xmin>'+bndbox[i][0]+'</xmin>')
            xml_file.write('    '+'<ymin>'+bndbox[i][1]+'</ymin>')
            xml_file.write('    '+'<xmax>'+bndbox[i][2]+'</xmax>')
            xml_file.write('    '+'<ymax>'+bndbox[i][3]+'</ymax>')
            xml_file.write('   '+'</bndbox>')
            xml_file.write('  '+'</object>')
        xml_file.write('</annotation>')
        xml_file.close() 

#creating datalist
def parse_xml(path):
    
    tree = ET.parse(path)
    root = tree.getroot()
    
    label = list()
    truncate  = list()
    occlusion = list()
    bndbox    = list() 
    
    for object in root.iter('object'):
        obj_label = object.find('name').text.lower()
       
        if obj_label not in label_map:
            continue
        
        obj_truncate = object.find('truncate').text
        obj_occlusion = object.find('occlusion').text    
        
        box   = object.find('bndbox')
        xmin  = box.find('xmin').text
        ymin  = box.find('ymin').text
        xmax  = box.find('xmax').text
        ymax  = box.find('ymax').text
        
        label.append(label_map[obj_label])
        truncate.append(obj_truncate)
        occlusion.append(obj_occlusion)
        bndbox.append([xmin,ymin,xmax,ymax])
        
    return {'label':label,'truncate':truncate,'occlusion':occlusion,'bndbox':bndbox }
        
#saving as json format
def save_as_json(xml_source,img_source,outputdir):    
    file_name     = []
    train_objects = []
    img_dir       = []
    
    for root,dir,file in os.walk(xml_source):
        for name in file:
            file_name.append(name)
    for ind in file_name:
        objects = parse_xml(os.path.join(xml_source,ind))
        train_objects.append(objects)
        img_dir.append(os.path.join(img_source,ind))
   
    with open(os.path.join(outputdir,'Train_Objects.json'),'w') as k:
        json.dump(train_objects,k)
        
    with open(os.path.join(outputdir,'Train_Image.json'),'w') as k:
        json.dump(img_dir,k)
            
    with open(os.path.join(outputdir,'Label_Map.json'),'w') as j:
        json.dump(label_map,j)
 
#intersection with bounding box presentation
def intersection(set1,set2):
    lower_bound = torch.max(set1[ : , :2 ].unsqueeze(1) , set[ : , :2 ].unsqueeze(0)) #n1,n2,2
    upper_bound = torch.min(set1[ : , 2: ].unsqueeze(1) , set[ : , 2: ].unsqueeze(0))
    
    intersect   = torch.clamp(upper_bound-lower_bound , 0) #n1,n2,2
    
    return intersect[: , : , 0 ] * intersect[ : , : , 1 ] #n1,n2

#intersection over union with bounding box presentation           
def iou(set1,set2):
    area_set1 = set1[ : , :2 ] - set1[ : , 2: ]
    area_set1 = area_set1[ : , 0 ] * area_set1[ : , 1 ]
    
    area_set2 = set2[ : , :2 ] - set2[ : , 2: ]
    area_set2 = area_set2[ : , 0 ] * area_set2[ : , 1 ]
    
    sum_of_area     = area_set1.unsqueeze(1) + area_set2.unsqueeze(0)
    intersect       = intersection(set1 , set2)
    union           = sum_of_area - intersect
    
    return  intersect / union
    
def decimate(t,i):
    for d in t.dim():
        if not i[d] == None :
            tensor = t.index_select(dim = d,
                                    index = torch.arange(start = 0, end = t.size(d), step = i).long())
    return tensor
    
    