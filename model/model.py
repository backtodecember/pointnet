import torch
import torch.nn as nn
import torch.nn.functional as F


class transformationNet(nn.Module):
    def __init__(self,input_d = 3):
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
        x = nn.MaxPool1d(N_pts,stride = 1).view(batch_size,a)#Bx1024 
        x = self.block4(x) #Bx512
        x = self.block5(x) #Bx256
        x = self.block6(x) #Bx input_d**2 (row major of matrix)

        indentity_matrix = torch.eye(self.input_d).view(self.input_d**2)
        x += indentity_matrix
        return x.view(-1,self.input_d,self.input_d)


class PointNetClassification(nn.Module):
    def __init__(self,N_classes):
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
        """
        _,N_pts,_ = x.shape
        transform = self.transform1(x) #B*3*3
        x = torch.bmm(x,transform) #BxNx3
        x.transpose(2,1) #Bx3xN
        x = self.block1(x) #Bx64xN
        x = self.block2(x) #Bx64xN
        x.transpose(2,1) #BxNx64
        transform = self.transform2(x) #B*64*64
        x = torch.bmm(x,transform) #BxNx64
        x.transpose(2,1) #Bx64xN
        x = self.block3(x) #Bx64xN
        x = self.block4(x) #Bx128xN
        x = self.block5(x) #Bx1024xN
        batch_size,a,N_pts = x.shape
        x = nn.MaxPool1d(N_pts,stride = 1).view(batch_size,a)#Bx1024 
        x = self.block6(x) #Bx512 
        x = self.block7(x) #Bx256
        x = self.block8(x) #BxN_classes
        
        return x

