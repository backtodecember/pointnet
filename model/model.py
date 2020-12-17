import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class transformationNet(nn.Module):
    def __init__(self,input_d = 3):
        super(transformationNet, self).__init__()
        self.input_d = input_d
        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_d,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU())
        self.block4 = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.block5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.block6 = nn.Sequential(
            nn.Linear(256, self.input_d**2))
        


    def forward(self,x):
        """
        Parameter:
        ------------
        x is of size BxNxd

        Return:
        ------------
        - Bxinputxinput, the transformation matrix
        """

        #transpose x 
        x = x.transpose(2,1) #Bx input_d xN
        x = self.block1(x) #Bx64xN
        x = self.block2(x) #Bx128xN
        x = self.block3(x) #Bx1024xN
        batch_size,a,N_pts = x.shape
#         pool = nn.MaxPool1d(N_pts,stride = 1)
#         x = pool(x).view(batch_size,a)#Bx1024 
        x = torch.max(x,dim=2).values
        x = self.block4(x) #Bx512
        x = self.block5(x) #Bx256
        x = self.block6(x) #Bx input_d**2 (row major of matrix)

        indentity_matrix = torch.eye(self.input_d).view(self.input_d**2).to(device)
        x += indentity_matrix
        
        return x.view(-1,self.input_d,self.input_d)


class PointNetClassification(nn.Module):
    def __init__(self,N_classes):
        super(PointNetClassification, self).__init__()
        self.transform1 = transformationNet(3)
        self.block1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.transform2 = transformationNet(64)
        self.block3 = nn.Sequential(
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.block4 = nn.Sequential(
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.block5 = nn.Sequential(
            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU())
        self.block6 = nn.Sequential(
            nn.Linear(1024,512),
            nn.Dropout(0.7),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.block7 = nn.Sequential(
            nn.Linear(512,256),
            nn.Dropout(0.7),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.block8 = nn.Sequential(
            nn.Linear(256,N_classes))


    def forward(self,x):
        """
        Parameter:
        ------------
        x is of size BxNxd
        
        Return:
        ------------
        Predictions
        Feature matrices
        """
        _,N_pts,_ = x.shape
        transforms_1 = self.transform1(x) #B*3*3
        x = torch.bmm(x,transforms_1) #BxNx3
        x = x.transpose(2,1) #Bx3xN
        x = self.block1(x) #Bx64xN
        
        x = self.block2(x) #Bx64xN
        x = x.transpose(2,1) #BxNx64
        transforms_2 = self.transform2(x) #B*64*64
        x = torch.bmm(x,transforms_2) #BxNx64
        x = x.transpose(2,1) #Bx64xN
        x = self.block3(x) #Bx64xN
        x = self.block4(x) #Bx128xN
        x = self.block5(x) #Bx1024xN
        batch_size,a,N_pts = x.shape
#         pool = nn.MaxPool1d(N_pts,stride = 1)
#         x = pool(x).view(batch_size,a)#Bx1024
        x = torch.max(x,dim=2).values
        x = self.block6(x) #Bx512 
        x = self.block7(x) #Bx256
        x = self.block8(x) #BxN_classes
        
        return x,transforms_2

##This can be used for both part segmentation and semantic segmentation
class PointNetDenseClassification(nn.Module):
    def __init__(self,N_classes):
        super(PointNetDenseClassification, self).__init__()
        self.transform1 = transformationNet(3)
        self.block1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.transform2 = transformationNet(64)
        self.block3 = nn.Sequential(
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.block4 = nn.Sequential(
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.block5 = nn.Sequential(
            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU())
        
        #local features
        self.block6 = nn.Sequential(
            nn.Conv1d(1088,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.block7 = nn.Sequential(
            nn.Conv1d(512,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.block8 = nn.Sequential(
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.block9 = nn.Sequential(
            nn.Conv1d(128,N_classes,1),
            nn.BatchNorm1d(N_classes),
            nn.ReLU())

    def forward(self,x):
        """
        Parameter:
        ------------
        x is of size BxNxd
        
        Return:
        ------------
        Predictions
        Feature matrices
        """
        _,N_pts,_ = x.shape
        transforms_1 = self.transform1(x) #B*3*3
        x = torch.bmm(x,transforms_1) #BxNx3
        x = x.transpose(2,1) #Bx3xN
        x = self.block1(x) #Bx64xN
        
        x = self.block2(x) #Bx64xN
        x = x.transpose(2,1) #BxNx64
        transforms_2 = self.transform2(x) #B*64*64
        x = torch.bmm(x,transforms_2) #BxNx64
        b = x.transpose(2,1) #Bx64xN
        
        x = self.block3(b) #Bx64xN
        x = self.block4(x) #Bx128xN
        x = self.block5(x) #Bx1024xN
        batch_size,a,N_pts = x.shape
        #extracted global feature
        x = torch.max(x,dim=2).values#Bx1024
        
        #now compute local features
        x = x.unsqueeze(2) #Bx1024x1
        tmp = copy.copy(x)
        for i in range(N_pts-1):
            x = torch.cat((x,tmp),dim =2)#Bx1024xN
        
        c = torch.cat((b,x),dim = 1)#Bx1088xN
        c = self.block6(c) #Bx512xN
        c = self.block7(c) #Bx256xN
        c = self.block8(c) #Bx128xN
        c = self.block9(c) #BxN_classesxN
        c = c.transpose(2,1) #BxNxN_classes
        
        return c,transforms_2

