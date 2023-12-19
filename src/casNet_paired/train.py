def merge_images3(sources, targets, truth, k=10, displayed_images=9):###with ground truth
    import numpy as np
    _, _, h, w = sources.shape
    row = int(np.ceil(np.sqrt(displayed_images)))
    merged = np.zeros([3, row * h, row * w * 3])
    for idx, (s, t,gt) in enumerate(zip(sources, targets,truth)):
      i = idx // row
      j = idx % row
      merged[:, i * h:(i + 1) * h, (j * 3) * w:(j * 3 + 1) * w] = s
      merged[:, i * h:(i + 1) * h, (j * 3 + 1) * w:(j * 3 + 2) * w] = t
      merged[:, i * h:(i + 1) * h, (j * 3 + 2) * w:(j * 3 + 3) * w] = gt
    return merged.transpose(1, 2, 0)   

from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
def test_g2(modelt,epoch=-1,acc=-1,best_acc=0,best_epoch=0):
    import numpy as np
    import torch
    from torchvision import datasets, transforms, models
    import torch.nn.functional as F    
    import os
    import scipy.io
    import numpy as np
    import scipy.misc
    from torch import nn
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg

    
    criterion = nn.NLLLoss()   
    displayed_images=9
    image_width=256
    image_height=256
    
    fixed_clean=torch.tensor(np.ones((displayed_images,3,image_height,image_width)))
    fixed_noisy=torch.tensor(np.ones((displayed_images,3,image_height,image_width)))
    fixed_fake=torch.tensor(np.ones((displayed_images,3,image_height,image_width)))
    
    test_noisy_dir="/data/private/Radar_data/mD_sr/Data_vsr_test"
    #"/data/private/Radar_data/Range_azimoth/test"
    #test_clean_dir="/no_backups/s1411/Denoising/clean_test"
    
    #classifier_path = os.path.join('/no_backups/s1411/Gait_classifier2/models/classifiermodel.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #model = torch.load(classifier_path)
    #model.eval()
    #model.to(device)
    ###################noisy
    test_transforms = transforms.Compose([#transforms.Resize(256),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ])
    
    batch_size2=32
    test_data2 = datasets.ImageFolder(test_noisy_dir, transform=test_transforms)
    testloader_noisy = torch.utils.data.DataLoader(test_data2, shuffle=False, batch_size=batch_size2)
    #test_data = datasets.ImageFolder(test_clean_dir, transform=test_transforms)
    ##testloader_clean = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)   
    
    # print(testloader_noisy.dataset.class_to_idx)
    # print(testloader_clean.dataset.class_to_idx)

    #######################g21 on noisydata


##plt.imshow(((inputs.permute(1, 2, 0)*0.5+0.5)*255).numpy().astype(np.uint8)) ##### 255

    modelt.eval()
    modelt.to(device)

    test_loss3 = 0
    accuracy3 = 0
    
    mse1=0#best 0
    best_mse=1000000
    PSNR1=0  ##best INF
    SSIM1,SSIM2=0,0 ##best(1.0,1.0)
    UQI1= 0 ## best 1
    VIF1 = 0 ## best 1


    with torch.no_grad():
        i=0
        c=0
        for inputs, labels in testloader_noisy:
         
            #inputs=transforms.functional.crop(inputs,50,115,image_height,image_width)
            inputs, labels = inputs.to(device), labels.to(device)
            clean_ref = inputs[:,:,:,0:256].to(device)
            noisy = inputs[:,:,:,256:512].to(device)
            fake=modelt(noisy).detach()
            paired_img=((clean_ref.cpu().permute(0,2, 3, 1)*0.5+0.5)*255).numpy().astype(np.uint8)
            fake_img=((fake.cpu().permute(0,2, 3, 1)*0.5+0.5)*255).numpy().astype(np.uint8)
            if i==2 or i==8 or i==12 or i==16 or i==20 or i==24 or i==28 or i==32 or i==36: 
                fixed_noisy[c]=noisy[1][None,:,:,:]
                fixed_fake[c]=fake[1][None,:,:,:]
                fixed_clean[c]=clean_ref[1][None,:,:,:]
                c+=1
            mse1+=mse(fake_img,paired_img)
            PSNR1+=psnr(fake_img,paired_img)
            ssim_tmp1,ssim_tmp2,vif_tmp=0,0,0
            for t in range(10):
                x1,x2=ssim(fake_img[t],paired_img[t])
                x3=vifp(fake_img[t],paired_img[t])
                ssim_tmp1+=x1
                ssim_tmp2+=x2
                vif_tmp+=x3
            SSIM1+=ssim_tmp1/10
            SSIM2+=ssim_tmp2/10
            UQI1+=uqi(fake_img,paired_img)
            VIF1+=vif_tmp/10
            i += 1
            if i == 37:
                break
           
    msa1=mse1/i            
    PSNR1=PSNR1/i            
    SSIM1=SSIM1/i            
    SSIM2=SSIM2/i            
    UQI1=UQI1/i            
    VIF1=VIF1/i            
    print(f"i: {i:.4f}.. "
          f"MSE'0': {mse1:.4f}.. "
          f"PSNR'inf': {PSNR1:.4f}.. "
          f"SSIM1'1': {SSIM1:.4f}.. "
          f"SSIM2'1': {SSIM2:.4f}.. "
          f"UQI'1': {UQI1:.3f}.."
          f"VIF'1': {VIF1:.3f}",flush=True)
          
    sample_path_n2c='./samples2/n2c'
    merged = merge_images3(fixed_noisy, fixed_fake, fixed_clean)   
    os.makedirs(sample_path_n2c, exist_ok = True)
    path = os.path.join(sample_path_n2c, 'sample-%d-n-c.png' % (epoch))
    scipy.misc.imsave(path, merged)
    f = plt.figure()
    plt.ioff()
    plt.axis("off")
    plt.title(" Epoch: "+ str(epoch)+" Avg_MSE:"+str("%.4f" % mse1)+" PSNR:"+str("%.4f" % PSNR1)+" SSIM1:"+str("%.4f" % SSIM1)+" SSIM2:"+str("%.4f" % SSIM2)+" UQI:"+str("%.4f" % UQI1)+" VIF:"+str("%.4f" % VIF1), fontsize=4)
    plt.imshow(mpimg.imread(path))
    path = os.path.join(sample_path_n2c, 'sample-%d-n-c.pdf' % (epoch))
    plt.savefig(path)#,bbox_inches='tight',dpi=300)
    print('saved %s' % path,flush=True)
    modelt.train()
    return mse1,PSNR1,SSIM1,SSIM2,UQI1,VIF1
    
def genOut(modelt,displayed_images=1):
    import numpy as np
    import torch
    from torchvision import datasets, transforms, models
    import torch.nn.functional as F    
    import os
    import scipy.io
    import numpy as np
    import scipy.misc
    from torch import nn
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg

    
    image_width=256
    image_height=256
    
    fixed_clean=torch.tensor(np.ones((displayed_images,3,image_height,image_width)))
    fixed_noisy=torch.tensor(np.ones((displayed_images,3,image_height,image_width)))
    fixed_fake=torch.tensor(np.ones((displayed_images,3,image_height,image_width)))
    
    test_noisy_dir="/data/private/Radar_data/mD_sr/Data_vsr_test"
    #"/data/private/Radar_data/Range_azimoth/test"    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    ###################noisy
    test_transforms = transforms.Compose([#transforms.Resize(256),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ])
    
    batch_size=1
    test_data2 = datasets.ImageFolder(test_noisy_dir, transform=test_transforms)
    testloader_noisy = torch.utils.data.DataLoader(test_data2, shuffle=False, batch_size=batch_size)
    
    modelt.eval()
    modelt.to(device)

    with torch.no_grad():
        i=0
        for inputs, labels in testloader_noisy:
            #inputs=transforms.functional.crop(inputs,50,115,image_height,image_width)
            inputs, labels = inputs.to(device), labels.to(device)
            fixed=inputs[:,:,:,256:512]
            clean_ref = inputs.to(device)[:,:,:,0:256]
            fake=modelt(fixed).detach()
            fixed_fake[0]=fake
            fake_img=((fake[0].cpu().permute(1, 2, 0)*0.5+0.5)*255).numpy().astype(np.uint8)
            ref_img=((clean_ref[0].cpu().permute(1, 2, 0)*0.5+0.5)*255).numpy().astype(np.uint8)

            sample_path_n2c='./gen_out'
            sample_path_c='./gen_out_c'
            path = os.path.join(sample_path_n2c, 'sample-%d-n-c.png' % (i))
            scipy.misc.imsave(path, fake_img)
            path2 = os.path.join(sample_path_c, 'sample-%d-n-c.png' % (i))
            scipy.misc.imsave(path2, ref_img)

            i += 1
    return 0    
def prog():
  import os
  import pathlib
  import shutil
  import time
  import datetime
  from torchsummary import summary
  import torch.nn as nn
  import torch.nn.functional as F

  import torch
  import torchvision

  from torchvision import datasets, transforms, models

  from matplotlib import pyplot as plt
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter()

  import pickle
  import scipy.io
  import numpy as np
  import scipy.misc
  import random
  random.seed(42)
  from PIL import Image
  from matplotlib import cm
  import matplotlib.image as mpimg
  
  import imageio  
  
  import sys
  sys.path.insert(1,'/no_backups/s1411/models')
  import attention_unet_plusplus
  from attention_unet_plusplus import AttUNetPlusPlus

  image_size=256
  g_conv_dim=64####################
  d_conv_dim=64
  n_residual_blocks=6
  input_shape=(3,image_size,image_size)
  use_reconst_loss = True # to avoid
  

  train_iters=10000000   #260 epoch 
  batch_size=1
  displayed_images=9

  num_workers=2
  lr=0.0007
  beta1=0.5
  beta2=0.999
  
  n_epochs_constlr=100
  n_epochs_decay=100 # total 240epochs
  end_epoch=200
  
  model_path='./models'
  sample_path_c2n='./samples/c2n'
  sample_path_n2c='./samples/n2c'

  train_path="/data/private/Radar_data/mD_sr/Data_vsr"
  #"/data/private/Radar_data/Range_azimoth/train"
  #noisy_path="/no_backups/s1411/Denoising/noisy_train"
  #clean_testp="/no_backups/s1411/Denoising/clean_test"
  noisy_testp="/data/private/Radar_data/mD_sr/Data_vsr_test"
  #"/data/private/Radar_data/Range_azimoth/test"

  
  log_step=200
  
  sample_epoch=9
  metric_epoch=10
  save_epoch=60 ## 240
  

    
  
  class Discriminator_int(nn.Module):
        def __init__(self, input_shape):
            super(Discriminator_int, self).__init__()

            channels, height, width = input_shape

            # Calculate output shape of image discriminator (PatchGAN)
            self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
            

            def discriminator_block(in_filters, out_filters, normalize=True):
                """Returns downsampling layers of each discriminator block"""
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
                if normalize:
                    layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.conv1=nn.Conv2d(channels, 64, 4, stride=2, padding=1)
            self.conv2=nn.Conv2d(64, 128, 5, stride=2, padding=2)
            self.nrm=nn.InstanceNorm2d(128)
            self.conv3=nn.Conv2d(128, 256, 4, stride=2, padding=1)
            self.nrm2=nn.InstanceNorm2d(256)
            self.conv4=nn.Conv2d(256, 512, 4, stride=1, padding=1)
            self.nrm3=nn.InstanceNorm2d(512)
            self.zero=nn.ZeroPad2d((1, 0, 1, 0))
            self.conv5=nn.Conv2d(512, 1, 4, padding=1)
            

        def forward(self, x):
          intermediate_results = {}        
          x = nn.LeakyReLU(0.2, inplace=True)(self.conv1(x))         
          x = intermediate_results["block1"] = self.conv2(x)
          x = nn.LeakyReLU(0.2, inplace=True)(self.nrm(x))
          x = intermediate_results["block2"] = self.conv3(x)
          x = nn.LeakyReLU(0.2, inplace=True)(self.nrm2(x))
          x = intermediate_results["block3"] = self.conv4(x)
          x = nn.LeakyReLU(0.2, inplace=True)(self.nrm3(x))
          #x = self.zero(x)
          res = self.conv5(x)
          
          return res, intermediate_results

  class Discriminator(nn.Module):
        def __init__(self, input_shape):
            super(Discriminator, self).__init__()

            channels, height, width = input_shape

            # Calculate output shape of image discriminator (PatchGAN)
            self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

            def discriminator_block(in_filters, out_filters, normalize=True):
                """Returns downsampling layers of each discriminator block"""
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
                if normalize:
                    layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *discriminator_block(channels, 64, normalize=False),
                *discriminator_block(64, 128),
                *discriminator_block(128, 256),
                *discriminator_block(256, 512),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(512, 1, 4, padding=1)
            )

        def forward(self, img):
            return self.model(img)
            


  def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d): ##############################################################################CHANGED
      nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find("BatchNorm2d") != -1:
      nn.init.normal_(m.weight.data, 1.0, 0.02)
      nn.init.constant_(m.bias.data, 0)


  def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
      x = x.cuda()
    return torch.autograd.Variable(x)


  def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
      x = x.cpu()
    return x.data.numpy()


  # def reset_grad():
    # """Zeros the gradient buffers."""
    # g_optimizer.zero_grad()
    # d_optimizer.zero_grad()
  def reset_grad():
    """Zeros the gradient buffers."""
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()
    #d2_optimizer.zero_grad()
    
 
  def merge_images(sources, targets, k=10, displayed_images=displayed_images): ###unpaired data used >> no ground truth
    _, _, h, w = sources.shape
    row = int(np.ceil(np.sqrt(displayed_images)))
    merged = np.zeros([3, row * h, row * w * 2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
      i = idx // row
      j = idx % row
      merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
      merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged.transpose(1, 2, 0) 

  class ReplayBuffer:
        def __init__(self, max_size=50):
            assert max_size > 0
            self.max_size = max_size
            self.data = []

        def push_and_pop(self, data):
            to_return = []
            for element in data.data:
                element = torch.unsqueeze(element, 0)
                if len(self.data) < self.max_size:
                    self.data.append(element)
                    to_return.append(element)
                else:
                    if random.uniform(0, 1) > 0.5:
                        i = random.randint(0, self.max_size - 1)
                        to_return.append(self.data[i].clone())
                        self.data[i] = element
                    else:
                        to_return.append(element)
            return torch.autograd.Variable(torch.cat(to_return))
            
            
  batch=36
  import torchvision.datasets as dset


  if torch.cuda.is_available():
      print("The code will run on GPU.")
  else:
      print("The code will run on CPU.")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  noisy_ds = dset.ImageFolder(root=noisy_testp,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))


  noisy_loader = torch.utils.data.DataLoader(dataset=noisy_ds,
                                                batch_size=batch,
                                                shuffle=False,
                                                num_workers=num_workers)
  
  #clean_iter = iter(clean_loader)
  noisy_iter = iter(noisy_loader)
  # fixed data for sampling
  #fixed_clean=torch.tensor(np.ones((displayed_images,3,image_size,image_size)))
  fixed_noisy=torch.tensor(np.ones((displayed_images,3,image_size,image_size)))

  if displayed_images<=batch:
      #fixed_clean = clean_iter.next()[0][50:50+displayed_images]  #clean_iter return list first element is the batch of images second element is the label
      i=0
      while(i<displayed_images):
        fixed_noisy[i] = noisy_iter.next()[0][0,:,:,256:512]
        i+=1
  else:
      t=displayed_images//batch
      i=0
      while(i <=t):
          if i == t :
           # fixed_clean[i*batch:displayed_images] = clean_iter.next()[0][0:displayed_images-i*batch]
            fixed_noisy[i*batch:displayed_images] = noisy_iter.next()[0][0:displayed_images-i*batch,:,:,256:512]
          else:  
            #fixed_clean[i*batch:(i+1)*batch] = clean_iter.next()[0][0:batch]
            fixed_noisy[i*batch:(i+1)*batch] = noisy_iter.next()[0][0:batch,:,:,256:512]
          i=i+1
  
  print(fixed_noisy.shape)
  
  train_ds = dset.ImageFolder(root=train_path,
                             transform=transforms.Compose([
                                 #transforms.Resize(256),
                                 #transforms.CenterCrop(256),
                                 #transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))


  train_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)

  
  
  from Conformer.conformer_byMe import dense_encoder 
  from Conformer.conformer_byMe import dense_decoder
  from Conformer.conformer.modules.conformer_block import ConformerBlock 
  from Conformer.conformer_byMe import StandardUnit
  
  class Conformer_Gen(nn.Module):
        def __init__(self,width=64,Num_conf_blks=4):
          super(Conformer_Gen, self).__init__()
          self.enc=dense_encoder()
          self.conf_blks = nn.ModuleList([])
          for i in range(Num_conf_blks*2):
            self.conf_blks.append(ConformerBlock(heads=4,embed_size=width,feed_forward_dropout_p=0.2,attention_dropout_p=0.2,conv_dropout_p=0.1))
          self.dec=dense_decoder()
          self.ta=torch.tanh
        def forward(self, input):
          input=self.enc(input) ##  torch.Size([2, 64, 256, 127])  [B C T F']>>>.permute(0, 3, 2, 1)>>(2,127,256,64) [B F' T C]
          b, c, dim2, dim1 = input.shape #dim1 da el F'  [B C T F']
          out2=input
          for i in range(len(self.conf_blks)//2):
            out2=out2.permute(0, 3, 2, 1).contiguous().view(b*dim1, dim2, -1) #[BF' T C] [254, 256, 64] 
            out1=self.conf_blks[2*i](out2)
            out1=out1+out2
            out1=out1.view(b,dim1,dim2,-1).permute(0,2,1,3).contiguous().view(b*dim2,dim1,-1)  #[BT F' C] [512, 127, 64] 
            out2=self.conf_blks[2*i+1](out1)
            out2=out2+out1
            out2=out2.view(b,dim2,dim1,-1).permute(0,3,1,2).contiguous() ## [B C T F']
          out2=self.dec(out2)
          return self.ta(out2)
          
  class Conformer_Gen2(nn.Module):
        def __init__(self,width=128,Num_conf_blks=4):
          super(Conformer_Gen2, self).__init__()
          self.enc=dense_encoder(width = width//2)
          self.conf_blks = nn.ModuleList([])
          for i in range(Num_conf_blks*2):
            self.conf_blks.append(ConformerBlock(heads=4,embed_size=width,feed_forward_dropout_p=0.2,attention_dropout_p=0.2,conv_dropout_p=0.1))
          self.dec=dense_decoder(width = width)
          self.ta=torch.tanh
          
          self.enc2=dense_encoder(width = width//2)
          #self.conf_blks2 = nn.ModuleList([])
          #for i in range(Num_conf_blks*2):
          #  self.conf_blks2.append(ConformerBlock(heads=4,embed_size=width,feed_forward_dropout_p=0.2,attention_dropout_p=0.2,conv_dropout_p=0.1))
          #self.dec2=dense_decoder()
          
        def forward(self, inputs):
          inputs2=self.enc(inputs) ##  torch.Size([2, 64, 256, 127])  [B C T F']>>>.permute(0, 3, 2, 1)>>(2,127,256,64) [B F' T C]
          #b, c, dim2, dim1 = inputs2.shape #dim1 da el F'  [B C T F']
          #out2=inputs2
          
          inputs3=self.enc2(inputs.permute(0,1,3,2)) ##  torch.Size([2, 64, 256, 127])  [B C F T']>>>.permute(0, 3, 2, 1)>>(2,127,256,64) [B T' F C]
          #b, c, dim2, dim1 = inputs3.shape #dim1 da el F'  [B C T F']      
          inputs4 = torch.cat((inputs2,inputs3),dim=1)
          b, c, dim2, dim1 = inputs4.shape
          #print(inputs4.shape)
          for i in range(len(self.conf_blks)//2):
            out2=inputs4.permute(0, 3, 2, 1).contiguous().view(b*dim1, dim2, -1) #[BF' T C] [254, 256, 128] 
            #print(out2.shape)
            out1=self.conf_blks[2*i](out2)
            out1=out1+out2
            out1=out1.view(b,dim1,dim2,-1).permute(0,2,1,3).contiguous().view(b*dim2,dim1,-1)  #[BT F' C] [512, 127, 64] 
            out2=self.conf_blks[2*i+1](out1)
            out2=out2+out1
            out2=out2.view(b,dim2,dim1,-1).permute(0,3,1,2).contiguous() ## [B C T F']
          out2=self.dec(out2)
          return self.ta(out2)
          
  class Conformer_Gen3(nn.Module):
        def __init__(self,width=64,Num_conf_blks=4):
          super(Conformer_Gen3, self).__init__()
          self.enc=dense_encoder(width = width//2)
          self.conf_blks = nn.ModuleList([])
          for i in range(Num_conf_blks*2):
            self.conf_blks.append(ConformerBlock(heads=4,embed_size=width,feed_forward_dropout_p=0.2,attention_dropout_p=0.2,conv_dropout_p=0.1))
          self.dec=dense_decoder(width = width, out_channels=width)
          
          self.enc2=dense_encoder(width = width//2)
          self.conf_blks2 = nn.ModuleList([])
          for i in range(Num_conf_blks*2):
            self.conf_blks2.append(ConformerBlock(heads=4,embed_size=width,feed_forward_dropout_p=0.2,attention_dropout_p=0.2,conv_dropout_p=0.1))
          self.dec2=dense_decoder(width = width, out_channels=width)
          
          self.out=StandardUnit(in_channel=2*width, out_channel= 3)
          self.ta=torch.tanh
          

          
        def forward(self, inputs):
          inputs2=self.enc(inputs) ##  torch.Size([2, 64, 256, 127])  [B C T F']>>>.permute(0, 3, 2, 1)>>(2,127,256,64) [B F' T C]
          #b, c, dim2, dim1 = inputs2.shape #dim1 da el F'  [B C T F']
          #out2=inputs2
          
          inputs3=self.enc2(inputs.permute(0,1,3,2)) ##  torch.Size([2, 64, 256, 127])  [B C F T']>>>.permute(0, 3, 2, 1)>>(2,127,256,64) [B T' F C]
          #b, c, dim2, dim1 = inputs3.shape #dim1 da el F'  [B C T F']      
          inputs4 = torch.cat((inputs2,inputs3),dim=1)
          b, c, dim2, dim1 = inputs4.shape
          
          inputs5 = inputs4.permute(0, 1, 3, 2)
          b2, c2, dim22, dim12 = inputs5.shape
          
          #print(inputs4.shape)
          for i in range(len(self.conf_blks)//2):
            out2=inputs4.permute(0, 3, 2, 1).contiguous().view(b*dim1, dim2, -1) #[BF' T C] [254, 256, 128] 
            out3=inputs5.permute(0, 3, 2, 1).contiguous().view(b2*dim12, dim22, -1)  
                    
            #print(out2.shape)
            out1=self.conf_blks[2*i](out2)
            out4=self.conf_blks2[2*i](out3)
            
            out1=out1+out2
            out4=out4+out3
            
            out1=out1.view(b,dim1,dim2,-1).permute(0,2,1,3).contiguous().view(b*dim2,dim1,-1)  #[BT F' C] [512, 127, 64] 
            out4=out4.view(b2,dim12,dim22,-1).permute(0,2,1,3).contiguous().view(b2*dim22,dim12,-1)  #[BT F' C] [512, 127, 64] 
            
            out2=self.conf_blks[2*i+1](out1)
            out3=self.conf_blks2[2*i+1](out4)
            out2=out2+out1
            out3=out3+out4
            
            out2=out2.view(b,dim2,dim1,-1).permute(0,3,1,2).contiguous() ## [B C T F']
            out3=out3.view(b2,dim22,dim12,-1).permute(0,3,1,2).contiguous() ## [B C T F']
          #print(out2.shape)
          #print(out3.shape)  
          out2=self.dec(out2)
          out3=self.dec2(out3.permute(0, 1, 3, 2))
          outtemp = torch.cat((out2,out3),dim=1)
          out=self.out(outtemp)
          
          return self.ta(out)
          
  import unet_d
  from unet_d import UNet_g         
  class cas_unet(nn.Module):
    def __init__(self):
        super(cas_unet, self).__init__()
        self.modelA = UNet_g(3,32)
        self.modelB = UNet_g(32,32)
        self.modelC = UNet_g(32,3)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x1 = self.modelB(x1)
        x1 = self.modelC(x1)
        return x1         

          
    
  criterion = nn.CrossEntropyLoss()  # Awel part criterion(real_output,1)
  criterion_GAN = torch.nn.MSELoss()
  criterion_cycle = torch.nn.MSELoss()
  criterion_L1 = torch.nn.L1Loss()
  
  #g12_old = AttUNetPlusPlus(in_channel=3, ngf=g_conv_dim) ##to compare

  
  #g12 = AttUNetPlusPlus(in_channel=3, ngf=g_conv_dim) ##1 clean   2 noisy   g12 from 1 to 2 >> from clean to noisy
  
  g21 = cas_unet() #Conformer_Gen3(Num_conf_blks=4)
  
  d1 = Discriminator_int(input_shape) ## discrimenator ffor clean 
  #d2 = Discriminator(input_shape)
  #g12.apply(weights_init)
  g21.apply(weights_init)
  d1.apply(weights_init)
  #d2.apply(weights_init)
  
  #g21_path = "./models/g21-last.pkl"
 # g12_path = "/misc/data/private/Radar_data/Denoise_gait_unet++_style_total/models/g12-last.pkl"
  #d1_path = "./models/d1-last.pkl"
  #d2_path = "/misc/data/private/Radar_data/Denoise_gait_unet++_style_total/models/d2-last.pkl"

  #g21.load_state_dict(torch.load(g21_path))
  #for param in g21.parameters():
        #param.requires_grad_(True)
  #d1.load_state_dict(torch.load(d1_path))
  #for param in d1.parameters():
        #param.requires_grad_(True)
                
  #classifier_VGG_path = "/no_backups/s1411/Gait_classifier_train_vgg19_corrected/models/classifier_CN_updated_VGG19org.pth"
  

  #vgg = models.vgg19(pretrained=True).features

  #vgg.load_state_dict(torch.load(classifier_VGG_path))  
  #for param in vgg.parameters():
  #      param.requires_grad_(False)  
  #vgg.eval()



  g_optimizer = torch.optim.Adam(g21.parameters(), lr, [beta1, beta2])
  d_optimizer = torch.optim.Adam(d1.parameters(), lr, [beta1, beta2])
  #d2_optimizer = torch.optim.Adam(d2.parameters(), lr, [beta1, beta2])

  #def lambda_rule(epoch):
            #lr_l = 1.0 - max(0, epoch + 1 - n_epochs_constlr) / float(n_epochs_decay + 1)
            #return lr_l
  schedulerG = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.85)
  schedulerD = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=30, gamma=0.85)
  #schedulerD2 = torch.optim.lr_scheduler.StepLR(d2_optimizer, step_size=10, gamma=0.85)
              

  #schedulerG = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_rule)
  #schedulerD = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda_rule)
  #schedulerD2 = torch.optim.lr_scheduler.LambdaLR(d2_optimizer, lr_lambda=lambda_rule)
        
 
  fake_clean_buff = ReplayBuffer()
  #fake_noisy_buff = ReplayBuffer()
  
  if torch.cuda.is_available():
    #g12_old.cuda()
    #g12.cuda()
    g21.cuda()
    d1.cuda()
    #d2.cuda()
    #vgg.cuda()

    print("The code will run on GPU.etrf3o")
    
  from torchsummary import summary

  print("g21 Parameters")
  summary(g21,(3,256,256))
  print("Discriminator Parameters")
  summary(d1,(3,256,256))  
  def get_features(image, model, layers=None):
        if layers is None:
          layers = {'0':'conv1_1','5':'conv2_1',
              '10':'conv3_1',
              '19':'conv4_1',
              '21':'conv4_2',#content layer
              '28':'conv5_1'}
        features = {}
        x = image 
        for name, layer in enumerate(model):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x
        return features

  def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

  def style_transfer_loss(vgg, outputs, targets):

    style_weights = {'conv1_1': 0.50,
                 'conv2_1': 0.40,
                 'conv3_1': 0.25,
                 'conv4_1': 0.25,
                 'conv5_1': 0.25} 

    content_weight = 1e-2  
    style_weight = 1e9
    target_features = get_features(targets, vgg)
    output_features = get_features(outputs, vgg)
    
    content_loss = torch.mean((target_features['conv4_2'] - output_features['conv4_2'])**2)
    style_loss=0.0
    style_grams = {layer: gram_matrix(output_features[layer]) for layer in output_features}

    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss    
    return total_loss,content_loss, style_loss 
  
  def percep_loss(vgg, outputs, targets,criterion_percp = torch.nn.L1Loss()):

    lambda_c_vec = [0.6, 0.5, 0.6]
    
    res,inter_res=vgg(outputs)

    outputs_layer = [inter_res["block1"], inter_res["block2"],inter_res["block3"]]

    # put the images into the vgg net
    res2,inter_res2=vgg(targets)

    # collect the layers of the vgg nerwork
    targets_layer = [inter_res2["block1"], inter_res2["block2"],inter_res2["block3"]]
    content_weight = 5e4
    # initial content loss
    content_loss = 0.001 * criterion_percp(outputs,outputs)#torch.tensor([0.0], requires_grad=True)
    # calculate style loss
    for output_layer, target_layer, lambda_c in zip(outputs_layer, targets_layer, lambda_c_vec):
    
        _, d, h, w = target_layer.shape
        content_loss += lambda_c * criterion_percp(output_layer,target_layer)/(d * h * w)
        #print(criterion_percp(gram_1,gram_2))

    return content_loss * content_weight
  
  train_iter = iter(train_loader)
  #noisy_iter = iter(noisy_loader)  

  iter_per_epoch = len(train_iter)

  print(iter_per_epoch) 
    
  image_list12=[]##1 clean   2 noisy   g12 from 1 to 2 >> from clean to noisy
  image_list21=[]
  
  accl,lossl,stl,=[],[],[]

  acc_best=0
  acc=0
  acc_best_stp=0
  mse1,PSNR1,SSIM1,SSIM2,UQI1,VIF1=0,0,0,0,0,0
  import time
  start = time.process_time()
  epoch=0
  #for i in range(90):
      #schedulerG.step()
      #schedulerD.step()
      #schedulerD2.step()
      #epoch=epoch+1
  for step in range(train_iters + 1):
 
    # reset data_iter for each epoch
    
    if (step + 1) % ((iter_per_epoch*3)//5) == 0: #+1 to not enter during first cycle
      #noisy_iter = iter(noisy_loader)
      train_iter = iter(train_loader)
      schedulerG.step()
      schedulerD.step()
      #schedulerD2.step()
      epoch=epoch+1
      print('Step [%d/%d], loss_g: %.4f, loss_L1: %.4f, loss_GAN: %.4f, '
                          'percep_loss_clean: %.4f, loss_d1: %.4f, lr_d:%.5f, lr_g:%.5f ,Epoch: %.1f' 
                          %(step, train_iters, loss_G.data, loss_L1.data, 
                          loss_GAN.data, percep_loss_clean.data, loss_d1.data,d_optimizer.param_groups[0]["lr"], g_optimizer.param_groups[0]["lr"], epoch))
      
      print("End of epoch" +str(epoch) +'/'+ str(end_epoch)+" Time Taken:" +str(time.process_time() - start)+"        Accuracy_best:"+str(acc_best)+" ..Epoch:"+ str(acc_best_stp),flush=True)
      
      if (epoch%metric_epoch==0):
          #acc,losst=test_g(g21,step)
          #writer.add_scalar("accuracy/step", acc, step)

          mse1,PSNR1,SSIM1,SSIM2,UQI1,VIF1=test_g2(g21,epoch,acc,acc_best,acc_best_stp)
          writer.add_scalar("mse1/epoch", mse1, epoch)
          writer.add_scalar("PSNR1/epoch", PSNR1, epoch)
          writer.add_scalar("SSIM1/epoch", SSIM1, epoch)
          writer.add_scalar("SSIM2/epoch", SSIM2, epoch)
          writer.add_scalar("UQI1/epoch", UQI1, epoch)
          writer.add_scalar("VIF1/epoch", VIF1, epoch)
          #writer.add_scalar("accuracy/epoch", acc, epoch)
          #accl.append(acc)
          #lossl.append(losst)
          #stl.append(step)
          
          print("Last_savedmdl_epoch:"+str(epoch))
          g21_path = os.path.join(model_path, 'g21-last.pkl')
          d1_path = os.path.join(model_path, 'd1-last.pkl')
          torch.save(d1.state_dict(), d1_path)
          torch.save(g21.state_dict(), g21_path)          
            # save the sampled images
      if epoch % sample_epoch == 0:
          fake_clean=torch.tensor(np.ones((displayed_images,3,image_size,image_size)))
          #fake_noisy=torch.tensor(np.ones((displayed_images,3,image_size,image_size)))
          i=0
          while(i <displayed_images):
              t1 = g21(to_var(fixed_noisy[i][None, :, :, :]).float()).detach()
             # t2 = g12(to_var(fixed_clean[i][None, :, :, :]).float()).detach()
              fake_clean[i]=t1.cpu()
              #fake_noisy[i]=t2.cpu()
              i=i+1

          noisy = to_data(fixed_noisy)
          fake_clean = to_data(fake_clean)

          merged = merge_images(noisy, fake_clean ,fake_clean)
          path = os.path.join(sample_path_n2c, 'sample-%d-m-s.png' % (epoch))
          
          scipy.misc.imsave(path, merged)
          
          plt.figure()
          plt.ioff()
          plt.axis("off")
          plt.title("Iter: "+str(step)+" Epoch: "+ str(epoch))
          plt.imshow(mpimg.imread(path))
          plt.savefig(path,bbox_inches='tight',dpi=240)
          plt.cla()
          print('saved %s' % path)
          image_list21.append(imageio.imread(path))
          

          writer.flush()  
      if epoch % save_epoch == 0:
          # save the model parameters 
          print("savedmdl_epoch:"+str(epoch))
          g21_path = os.path.join(model_path, 'g21-epoch-%d.pkl' % (epoch))
          d1_path = os.path.join(model_path, 'd1-epoch-%d.pkl' % (epoch))
          torch.save(d1.state_dict(), d1_path)
          torch.save(g21.state_dict(), g21_path)
          
      
      if epoch == end_epoch:
        break    
        
      start = time.process_time()



    # load dataset
    train_smpl, _ = train_iter.next()
    clean_ref = to_var(train_smpl[:,:,:,0:256])
    noisy = to_var(train_smpl[:,:,:,256:512])
    ##########3
    comb=torch.cat([clean_ref,noisy],dim=0)
    t2=transforms.RandomRotation([-20,20])(transforms.RandomPerspective(p=0.4)(transforms.RandomHorizontalFlip(p=0.5)(comb)))
    clean_ref=t2[0,:,:,:][None,:,:,:]
    noisy=t2[1,:,:,:][None,:,:,:]
    ############
    reset_grad()
    
    #identity loss 34an el sora mtt8yr4 awy 
    fake_clean = g21(noisy)

    loss_L1_B = criterion_L1(fake_clean,clean_ref)
    loss_L1 =  loss_L1_B
    
    #GAN Lossa
    
    d1_fake,_=d1(fake_clean)
    
    valid_fh = to_var(torch.tensor(np.ones(d1_fake.shape))).float()
    fake_fh = to_var(torch.tensor(np.zeros(d1_fake.shape))).float()

    loss_GAN_21 = criterion_GAN(d1_fake,valid_fh) 
    
    loss_GAN = loss_GAN_21

    percep_loss_clean=percep_loss(d1,fake_clean,clean_ref)
    
    loss_G = 50 * loss_L1 + 0.005*percep_loss_clean + 1.0*loss_GAN  #######################################
    loss_G.backward()
    g_optimizer.step()
    
    
    #train D1
    reset_grad()
    
    d1_real2,_=d1(clean_ref)
    loss_real = criterion_GAN(d1_real2, to_var(torch.tensor(np.ones(d1_real2.shape))).float())
    
    #fake_clean_ = fake_clean_buff.push_and_pop(fake_clean)
    d1_fake2,_ =d1(fake_clean.detach())
    
    loss_fake = criterion_GAN(d1_fake2, to_var(torch.tensor(np.zeros(d1_fake2.shape))).float())
    
    loss_d1 = (loss_real+loss_fake)/2.0 #######################
    loss_d1.backward()
    d_optimizer.step()


    # print the log info
    if (step+1) % log_step == 0:
         writer.add_scalar("loss_G/step", loss_G.data, step)
         writer.add_scalar("loss_L1/step", loss_L1.data, step)
         writer.add_scalar("loss_GAN/step", loss_GAN.data, step)
         writer.add_scalar("percep_loss_clean/step", percep_loss_clean.data, step)
         writer.add_scalar("loss_d1/step", loss_d1.data, step)         


  g21_path = os.path.join(model_path, 'g21-last.pkl')
  d1_path = os.path.join(model_path, 'd1-last.pkl')
  torch.save(d1.state_dict(), d1_path)
  torch.save(g21.state_dict(), g21_path)
  
  test_g2(g21,step + 1)
  genOut(g21)
  import subprocess
  list_of_files = subprocess.run(['python', '-m','pytorch_fid','./gen_out_c','./gen_out'], capture_output=True, text=True)
  print(list_of_files.stdout)    
  print(accl)
  print(lossl)
  print(stl)



  imageio.mimsave(sample_path_c2n+'/movie_c2n.gif', image_list12,duration=1)

  imageio.mimsave(sample_path_n2c+'/movie_n2c.gif', image_list21,duration=1)
  writer.close()

if __name__ == '__main__':
  import torch
  torch.cuda.empty_cache()
  prog()
