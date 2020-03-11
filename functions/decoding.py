from utils.box_utils import create_prior_boxes,encxcy_to_cxcy,iou
import torchvision.transforms.functional as F
import torch

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

def decoding(no_classes,min_score,max_overlap,top_k,predicted_scores,predicted_location):
    pri_box     = create_prior_boxes()
    scores      = F.softmax(predicted_scores,dim=2)
    batch_size  = predicted_scores.size(0)
    
    all_boxes  = list()
    all_labels = list()
    all_score  = list()
    
    for i in range(batch_size):
        decoded_loc   = encxcy_to_cxcy(predicted_location[i], pri_box)
        
        for c in range(1, no_classes):
            class_score   = scores[i][:,c] #8732
            score_abv_min = class_score > min_score #8732
            class_score   = class_score[score_abv_min] #no: qualified
            class_dec_loc = decoded_loc[score_abv_min]
            
            overlap       = iou(class_dec_loc, class_dec_loc) #no: qualified,no:qualified (to check how much a box is
            #overlap to other boxes)           
            class_score,sort_id = class_score.sort(descending=True)
            class_dec_loc = class_dec_loc[sort_id]
            
            class_box     = list()
            class_label   = list()
            class_score   = list()
            
            suppress      = torch.zeros(class_dec_loc.size(0), dtype=torch.uint8).to(device)
            
            for box in range(suppress.size()):
                if suppress[box] == 1:
                    continue
                to_suppress = overlap[box] > max_overlap
                to_suppress = torch.tensor(to_suppress , dtype=torch.uint8)
                suppress    = torch.max(suppress,to_suppress)
                
                suppress[box] = 0 #we will set unsppress the current box even if the overlap is 1
                #me The way how NMS work is 1st find JO of all objects and suppress all the overlapping boes that is greater than 
                # throushold except the most possible obj location.The most possible obj loc is suppress[box] as we sorted class_dec_loc 
                # wrt scores and so the former the more possible there is an object after that we skip the loop if already suppressed.To 
                # clarify,let say obj 1,3,5 are overlapped than thres and result 1,1,1 to these and then we keep the mosly likely
                # so we got 0,1,1 and for 3,1 and 5,1 we skip these,we only need to find 3,5.
                
            suppress = torch.tensor(suppress,dtype = torch.bool)               
            class_box.append(decoded_loc[1-suppress])
            class_label.append(torch.LongTensor(predicted_scores[1-suppress].sum().item()*[c]).to(device))
            class_score.append(class_score[1-suppress])
       
        no_object = len(class_box)
        #if no object is found ,we'll set it as background     
        if no_object == 0:
            class_box.append(torch.FloatTensor([[0.,0.,1.,1.]]).to(device))
            class_label.append(torch.LongTensor([0]).to(device))
            class_score.append(torch.FloatTensro([0.]).to(device))
                
        #concentenate into a single tensor
        class_box   = torch.cat(class_box , dim=0)
        class_label = torch.cat(class_label , dim=0)
        class_score = torch.cat(class_score , dim=0)
        
        if no_object > top_k:
            class_score,sort_id = class_score.sort(descending=True)
            class_score = class_score[:top_k]
            class_label = class_label[sort_id][:top_k]
            class_box   = class_box[sort_id][:top_k]
            
        all_boxes.append(class_box)
        all_labels.append(class_label)
        all_score.append(class_score)
    
    return all_boxes,all_boxes,all_score