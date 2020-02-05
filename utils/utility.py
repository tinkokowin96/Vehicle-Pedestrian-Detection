import os
import json
import xml.etree.ElementTree as ET

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
def save_as_json(sourcedir,outputdir):    
    file_name = []
    train_objects = []
    for root,dir,file in os.walk(sourcedir):
        for name in file:
            file_name.append(name)
    for ind in file_name:
        objects = parse_xml(os.path.join(sourcedir,ind))
        train_objects.append(objects)
   
    with open(os.path.join(outputdir,'Train_Objects.json'),'w') as k:
        json.dump(train_objects,k)
            
    with open(os.path.join(outputdir,'Label_Map.json'),'w') as j:
        json.dump(label_map,j)
        
