class DirectionalConv0(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DirectionalConv0,self).__init__()


        self.pad1=nn.ZeroPad2d(padding=(2,2,0, 0))
        self.pool1=nn.AvgPool2d(kernel_size=(1,5),stride=1)
        self.pad2=nn.ZeroPad2d(padding=(0,0,1, 1))
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=(3,1),stride=1)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.pad1(x)
        x=self.pool1(x)
        x=self.pad2(x)
        x=self.conv1(x)
        x=self.bn(x)
        x=self.relu(x)
        return x
    
    
class DirectionalConv90(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DirectionalConv90,self).__init__()


        self.pad1=nn.ZeroPad2d(padding=(0,0,2, 2))
        self.pool1=nn.AvgPool2d(kernel_size=(5,1),stride=1)
        self.pad2=nn.ZeroPad2d(padding=(1,1,0, 0))
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=(1,3),stride=1)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.pad1(x)
        x=self.pool1(x)
        x=self.pad2(x)
        x=self.conv1(x)
        x=self.bn(x)
        x=self.relu(x)
        return x
    
class DirectionalConv45(nn.Module):
    def __init__(self,in_channels,out_channels):
        
        super(DirectionalConv45, self).__init__()
        self.conv1=nn.Conv2d(3*in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.padA=nn.ZeroPad2d(padding=(0, 4, 0, 4))
        self.padB=nn.ZeroPad2d(padding=(1, 3, 1, 3))
        self.padC=nn.ZeroPad2d(padding=(2, 2, 2, 2))
        self.padD=nn.ZeroPad2d(padding=(3, 1, 3, 1))
        self.padE=nn.ZeroPad2d(padding=(4, 0, 4, 0))

        self.padF=nn.ZeroPad2d(padding=(0, 0, 1, 0))
        self.padG=nn.ZeroPad2d(padding=(1, 0, 0, 0))
        
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        
        
    def forward(self,x):
        
        
        

        A=self.padA(x)
        B=self.padB(x)
        C=self.padC(x)
        D=self.padD(x)
        E=self.padE(x)

    
        SUM_ABCD=A+B+C+D
        
        
        F=SUM_ABCD[:,:,:-1,:]
        
        F=self.padF(F)
        
        G=SUM_ABCD[:,:,:,:-1]
        G=self.padG(G)
        H=A+B+C+D+E
        
        

        ALL=torch.cat( (F,H,G),1 )
        out=self.conv1(ALL)
                
                
        x=out[:,:,2:-2,2:-2]
               
                
                
        
        x=self.bn(x)
        x=self.relu(x)
        
        
        return x
class DirectionalConv_45(nn.Module):
    def __init__(self,in_channels,out_channels):
        
        super(DirectionalConv_45, self).__init__()
        self.conv1=nn.Conv2d(3*in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.padA=nn.ZeroPad2d(padding=(0, 4, 0, 4))
        self.padB=nn.ZeroPad2d(padding=(1, 3, 3, 1))
        self.padC=nn.ZeroPad2d(padding=(2, 2, 2, 2))
        self.padD=nn.ZeroPad2d(padding=(3, 1, 1, 3))
        self.padE=nn.ZeroPad2d(padding=(4, 0, 0, 4))


        self.padF=nn.ZeroPad2d(padding=(1, 0, 0, 0))
        self.padG=nn.ZeroPad2d(padding=(0, 0, 0, 1))
        
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        
        
    def forward(self,x):
        
        
        

        A=self.padA(x)
        B=self.padB(x)
        C=self.padC(x)
        D=self.padD(x)
        E=self.padE(x)

    
        SUM_ABCD=A+B+C+D
        
        
        F=SUM_ABCD[:,:,:,:-1]
        
        F=self.padF(F)
        
        G=SUM_ABCD[:,:,1:,:]
        G=self.padG(G)
        H=A+B+C+D+E
        
        

        ALL=torch.cat( (F,H,G),1 )
        out=self.conv1(ALL)
                
                
        x=out[:,:,2:-2,2:-2]
               
                
                
        
        x=self.bn(x)
        x=self.relu(x)
        
        
        return x
      
      

class DConv(nn.Module):
   

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._45=DirectionalConv45(in_channels,out_channels//4)
        self.__45=DirectionalConv_45(in_channels,out_channels//4)
        self._0=DirectionalConv0(in_channels,out_channels//4)
        self._90=DirectionalConv90(in_channels,out_channels//4)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x1=self._45(x)
        x2=self.__45(x)
        x3=self._0(x)
        x4=self._90(x)
        out=self.bn(torch.cat((x1,x2,x3,x4),1))
       
        return out
