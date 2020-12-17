import utils
import os
import trimesh
import numpy as np
import pickle
import json

class ModelNet40Dataset():
    def __init__(self,data_path):
        category_list = os.listdir(data_path) 
        self.train = {}
        self.test = {}

        pcds = []
        labels = []
        original_names = []
        for category in category_list:
            print(category)
            for fn in os.listdir(data_path+category+'/train/'):             
                class_ID = utils.ModelNet40_classnames.index(category)
                mesh = trimesh.load_mesh(data_path+category+'/train/'+fn)
                #1024 sampled pts uniformly on the surface
                pcd = trimesh.sample.sample_surface(mesh,1024)[0].astype(np.float32)
                #normal to unit sphere
                pcd = pcd - np.expand_dims(np.mean(pcd, axis=0),0)
                dist = np.max(np.sqrt(np.sum(pcd**2, axis=1)),0)
                pcd = pcd/dist
                
                pcds.append(pcd)
                labels.append(class_ID)
                original_names.append(fn)
                
        self.train['pcds'] = pcds
        self.train['labels'] = labels
        self.train['fns'] = original_names
        
        pcds = []
        labels = []
        original_names = []
        for category in category_list:
            print(category)
            for fn in os.listdir(data_path+category+'/test/'):
                class_ID = utils.ModelNet40_classnames.index(category)        
                mesh = trimesh.load_mesh(data_path+category+'/test/'+fn)
                #1024 sampled pts uniformly on the surface
                pcd = trimesh.sample.sample_surface(mesh,1024)[0].astype(np.float32)
                #normal to unit sphere
                pcd = pcd - np.expand_dims(np.mean(pcd, axis=0),0)
                dist = np.max(np.sqrt(np.sum(pcd**2, axis=1)),0)
                pcd = pcd/dist
                
                pcds.append(pcd)
                labels.append(class_ID)
                original_names.append(fn)
                
        self.test['pcds'] = pcds
        self.test['labels'] = labels
        self.test['fns'] = original_names

            
##The ShapeNet data is too large to load all at once. Read batch from files during training instead.
class ShapeNetDataset():
    def __init__(self,data_path,class_ID):
        self.data_path = data_path
        self.class_ID = class_ID
        self.object_folder = utils.ShapeNet_folderID[class_ID]
        self.train = {}
        self.test = {}

    def get_N_parts(self,class_ID):
        split_path = self.data_path + 'train_test_split/' + 'shuffled_train_file_list.json'
        split_file = json.load(open(split_path, 'r'))
        object_ID = utils.ShapeNet_folderID[class_ID]
        for i in range(len(split_file)):
            fn = split_file[i].split('/')
            if fn[1] == object_ID:
                label_path = self.data_path + object_ID + '/points_label/' + fn[2] +'.seg'  
                label = []
                with open(label_path, 'r') as f:
                    for line in f:
                        ls = int(line)
                        label.append(ls - 1) #original labels are not zero based
                return max(label) + 1
        
    def get_data(self,data_type,batch_size):
        pcds = []
        labels = []
        fns = []
        split_path = self.data_path + 'train_test_split/' + 'shuffled_' + data_type +'_file_list.json'
        split_file = json.load(open(split_path, 'r'))
        N = len(split_file)
        
        object_ID = self.object_folder
        total_N = 0
        while total_N < 32:
            ind = np.random.choice(N,1)[0]
            fn = split_file[ind].split('/')
            if fn[1] == object_ID:
                total_N += 1
                object_path = self.data_path + object_ID + '/points/' + fn[2] +'.pts'
                label_path = self.data_path + object_ID + '/points_label/' + fn[2] +'.seg'
                pcd = []
                label = []
                with open(object_path, 'r') as f:
                    for line in f:
                        ls = [float(a) for a in line.strip().split()]
                        pcd.append(ls)
                with open(label_path, 'r') as f:
                    for line in f:
                        ls = int(line)
                        label.append(ls - 1) #original labels are not zero based
                pcds.append(pcd)
                labels.append(label)
                fns.append(fn)
        return pcds,labels,fns
    
    def get_train_data(self,batch_size):
        return self.get_data('train',batch_size)
    
    def get_test_data(self,batch_size):
        return self.get_data('test',batch_size)
    
    def get_val_data(self,batch_size):
        return self.get_data('val',batch_size)
    
    def get_all_data(self,data_type):
        pcds = []
        labels = []
        fns = []
        split_path = self.data_path + 'train_test_split/' + 'shuffled_' + data_type +'_file_list.json'
        split_file = json.load(open(split_path, 'r'))
        N = len(split_file)
        indices = np.arange(N)
        object_ID = self.object_folder       
        for ind in indices:
            fn = split_file[ind].split('/')
            if fn[1] == object_ID:
                object_path = self.data_path + object_ID + '/points/' + fn[2] +'.pts'
                label_path = self.data_path + object_ID + '/points_label/' + fn[2] +'.seg'
                pcd = []
                label = []
                with open(object_path, 'r') as f:
                    for line in f:
                        ls = [float(a) for a in line.strip().split()]
                        pcd.append(ls)
                with open(label_path, 'r') as f:
                    for line in f:
                        ls = int(line)
                        label.append(ls - 1) #original labels are not zero based
                pcds.append(pcd)
                labels.append(label)
                fns.append(fn)
        return pcds,labels,fns
if __name__ == "__main__":
#     ### ModelNet ######
#     dataset = ModelNet40Dataset('datasets/ModelNet40/')
#     filehandler = open('ModelNet40ProcessedDataset', 'wb') 
#     pickle.dump(dataset, filehandler)
#     filehandler = open('ModelNet40ProcessedDataset', 'rb') 
#     ModelNet40Data = pickle.load(filehandler)


    ###ShapeNet #####
    dataset = ShapeNetDataset('datasets/ShapeNet/',6)
    pcds,labels,fns = dataset.get_all_data('test')