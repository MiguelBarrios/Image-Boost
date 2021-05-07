import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu):
        self.ngpu = ngpu
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride = 1, padding=4 ),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(32, inplace = True),            
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2,inplace=True)
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 4,stride =  2, padding = 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 4,stride =  2, padding = 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels = 8, out_channels =  4,kernel_size = 4,stride= 2,padding= 1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=4,out_channels= 1,kernel_size= 4,stride= 1,padding= 0, bias=False),
            ## nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(25,1),
            nn.Sigmoid()
        )
    def forward(self, input):
        out = self.main(input)
        out = torch.flatten(out,1)
        out = self.classifier(out)
        return out

def loadSavedModel(path):
    lr = 0.0002
    net = Generator(0)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    net.to(device)
    total_epochs = epoch
    net.eval()
    return net, optimizer



