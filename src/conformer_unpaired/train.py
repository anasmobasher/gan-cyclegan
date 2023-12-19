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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt    
    import matplotlib.image as mpimg

    
    criterion = nn.NLLLoss()   
    displayed_images=9
    image_width=256
    image_height=256
    
    fixed_clean=torch.tensor(np.ones((displayed_images,3,image_height,image_width)))
    fixed_noisy=torch.tensor(np.ones((displayed_images,3,image_height,image_width)))
    fixed_fake=torch.tensor(np.ones((displayed_images,3,image_height,image_width)))
    
    test_noisy_dir= "/data/private/Radar_data/mD_sr/Data_vsr_test"
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
    plt.savefig(path,bbox_inches='tight',dpi=300)
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    
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
            os.makedirs(sample_path_n2c, exist_ok = True)   
            os.makedirs(sample_path_c, exist_ok = True)

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
  
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter()

  import pickle
  import scipy.io
  import numpy as np
  import scipy.misc
  import random
  random.seed(42)
  from PIL import Image
  import matplotlib.image as mpimg
  
  import imageio  
  
  import sys
  sys.path.insert(1,'/data/private/Radar_data/models')

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
  
  os.makedirs(model_path, exist_ok = True)
  os.makedirs(sample_path_c2n, exist_ok = True)
  os.makedirs(sample_path_n2c, exist_ok = True)


  train_path="/data/private/Radar_data/mD_sr/Data_vsr"
  #"/data/private/Radar_data/Range_azimoth/train"
  #noisy_path="/no_backups/s1411/Denoising/noisy_train"
  #clean_testp="/no_backups/s1411/Denoising/clean_test"
  noisy_testp="/data/private/Radar_data/mD_sr/Data_vsr_test"
  #"/data/private/Radar_data/Range_azimoth/test"

  
  log_step=200
  
  sample_epoch=3
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


  def reset_grad():
    """Zeros the gradient buffers."""
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()
    d2_optimizer.zero_grad()
    
 
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
  fixed_clean=torch.tensor(np.ones((displayed_images,3,image_size,image_size)))
  fixed_noisy=torch.tensor(np.ones((displayed_images,3,image_size,image_size)))

  if displayed_images<=batch:
      #fixed_clean = clean_iter.next()[0][50:50+displayed_images]  #clean_iter return list first element is the batch of images second element is the label
      i=0
      while(i<displayed_images):
        img=noisy_iter.next()[0]
        fixed_noisy[i] = img[0,:,:,256:512]
        fixed_clean[i] = img[0,:,:,0:256]
        i+=1
  else:
      t=displayed_images//batch
      i=0
      while(i <=t):
          if i == t :
            img=noisy_iter.next()[0]
            fixed_clean[i*batch:displayed_images] = img[0:displayed_images-i*batch,:,:,0:256]
            fixed_noisy[i*batch:displayed_images] = img[0:displayed_images-i*batch,:,:,256:512]
          else:  
            img=noisy_iter.next()[0]          
            fixed_clean[i*batch:(i+1)*batch] = img[0:batch,:,:,0:256]
            fixed_noisy[i*batch:(i+1)*batch] = img[0:batch,:,:,256:512]
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

  
  
  from CMGAN.generator import TSCNet3
  import unet_d
  from unet_d import UNet_g         
  class cas_unet(nn.Module):
    def __init__(self):
        super(cas_unet, self).__init__()
        self.modelA = UNet_g(3,64)
        self.modelB = UNet_g(64,64)
        self.modelC = UNet_g(64,3)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x1 = self.modelB(x1)
        x1 = self.modelC(x1)
        return x1         

          
    
  criterion = nn.CrossEntropyLoss()  # Awel part criterion(real_output,1)
  criterion_GAN = torch.nn.MSELoss()
  criterion_cycle = torch.nn.MSELoss()
  criterion_identity = torch.nn.L1Loss()
  
  #g12_old = AttUNetPlusPlus(in_channel=3, ngf=g_conv_dim) ##to compare

  
  g12 = TSCNet3()#AttUNetPlusPlus(in_channel=3, ngf=g_conv_dim) ##1 clean   2 noisy   g12 from 1 to 2 >> from clean to noisy
  
  g21 = TSCNet3()
  
  d1 = Discriminator_int(input_shape) ## discrimenator ffor clean 
  d2 = Discriminator_int(input_shape)
  g12.apply(weights_init)
  g21.apply(weights_init)
  d1.apply(weights_init)
  d2.apply(weights_init)
  

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


  g_params = list(g12.parameters()) + list(g21.parameters())

  g_optimizer = torch.optim.Adam(g_params, lr, [beta1, beta2])
  d_optimizer = torch.optim.Adam(d1.parameters(), lr, [beta1, beta2])
  d2_optimizer = torch.optim.Adam(d2.parameters(), lr, [beta1, beta2])

  def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - n_epochs_constlr) / float(n_epochs_decay + 1)
            return lr_l

  schedulerG =  torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=20, gamma=0.89)
  schedulerD =  torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=20, gamma=0.89)
  schedulerD2 =  torch.optim.lr_scheduler.StepLR(d2_optimizer, step_size=20, gamma=0.89)
        
 
  fake_clean_buff = ReplayBuffer()
  fake_noisy_buff = ReplayBuffer()
  
  if torch.cuda.is_available():
    #g12_old.cuda()
    g12.cuda()
    g21.cuda()
    d1.cuda()
    d2.cuda()
    #vgg.cuda()

    print("The code will run on GPU.etrf3o")
    
  from torchsummary import summary

  print("g21 Parameters")
  #summary(g21,(3,256,256))
  print("Discriminator Parameters")
  #summary(d1,(3,256,256))  

  
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
  noisy_iter = iter(train_loader)  

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
  psnr_best=0
  #for i in range(90):
      #schedulerG.step()
      #schedulerD.step()
      #schedulerD2.step()
      #epoch=epoch+1
  for step in range(train_iters + 1):
 
    # reset data_iter for each epoch
    
    if (step + 1) % ((iter_per_epoch*3)//5) == 0: #+1 to not enter during first cycle
      noisy_iter = iter(train_loader)
      train_iter = iter(train_loader)
      schedulerG.step()
      schedulerD.step()
      schedulerD2.step()
      epoch=epoch+1
      print('Step [%d/%d], loss_g: %.4f, loss_identity: %.4f, loss_GAN: %.4f, '
                      'loss_cycle: %.4f, percep_loss_total: %.4f, loss_d1: %.4f, loss_d2: %.4f, lr_d:%.5f, lr_g:%.5f ,Epoch: %.1f' 
                      %(step+1, train_iters, loss_G.data, loss_identity.data, 
                      loss_GAN.data, loss_cycle.data, percep_loss_total.data, loss_d1.data, loss_d2.data,d2_optimizer.param_groups[0]["lr"], g_optimizer.param_groups[0]["lr"], epoch),flush=True)
      print("End of epoch" +str(epoch) +'/'+ str(end_epoch)+" Time Taken:" +str(time.process_time() - start)+"        Accuracy_best:"+str(psnr_best)+" ..Epoch:"+ str(acc_best_stp),flush=True)
     
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
          if psnr_best<PSNR1:
            psnr_best = PSNR1
            acc_best_stp = epoch
            print("$$$$$$$$$$$$$$$$$best_savedmdl_epoch:"+str(epoch) + "  psnr_best: " + str(psnr_best))
            g21_path = os.path.join(model_path, 'g21-best.pkl')
            g12_path = os.path.join(model_path, 'g12-best.pkl')          
            d1_path = os.path.join(model_path, 'd1-best.pkl')
            d2_path = os.path.join(model_path, 'd2-best.pkl')
            torch.save(d1.state_dict(), d1_path)
            torch.save(d2.state_dict(), d2_path)
            torch.save(g21.state_dict(), g21_path)          
            torch.save(g12.state_dict(), g12_path) 
          print("Last_savedmdl_epoch:"+str(epoch))
          g21_path = os.path.join(model_path, 'g21-last.pkl')
          g12_path = os.path.join(model_path, 'g12-last.pkl')          
          d1_path = os.path.join(model_path, 'd1-last.pkl')
          d2_path = os.path.join(model_path, 'd2-last.pkl')
          torch.save(d1.state_dict(), d1_path)
          torch.save(d2.state_dict(), d2_path)
          torch.save(g21.state_dict(), g21_path)          
          torch.save(g12.state_dict(), g12_path)          
            # save the sampled images
      if epoch % sample_epoch == 0:
          fake_clean=torch.tensor(np.ones((displayed_images,3,image_size,image_size)))
          fake_noisy=torch.tensor(np.ones((displayed_images,3,image_size,image_size)))
          i=0
          while(i <displayed_images):
              t1 = g21(to_var(fixed_noisy[i][None, :, :, :]).float()).detach()
              t2 = g12(to_var(fixed_clean[i][None, :, :, :]).float()).detach()
              fake_clean[i]=t1.cpu()
              fake_noisy[i]=t2.cpu()
              i=i+1

          noisy, fake_noisy = to_data(fixed_noisy), to_data(fake_noisy)
          clean, fake_clean = to_data(fixed_clean), to_data(fake_clean)

          merged = merge_images(noisy, fake_clean ,fake_clean)
          path = os.path.join(sample_path_n2c, 'sample-%d-m-s.png' % (epoch))
          
          scipy.misc.imsave(path, merged)
          
          f = plt.figure()
          plt.ioff()
          plt.axis("off")
          plt.title("Iter: "+str(step)+" Epoch: "+ str(epoch))
          plt.imshow(mpimg.imread(path))
          plt.savefig(path,bbox_inches='tight',dpi=240)
          plt.cla()
          print('saved %s' % path)
          image_list21.append(imageio.imread(path))
          
          
          merged = merge_images(clean, fake_noisy, fake_noisy)      
          path = os.path.join(sample_path_c2n, 'sample-%d-s-m.png' % (epoch))
          
          scipy.misc.imsave(path, merged)
          f = plt.figure()
          plt.ioff()
          plt.axis("off")
          plt.title("Iter: "+str(step)+" Epoch: "+ str(epoch))
          plt.imshow(mpimg.imread(path))
          plt.savefig(path,bbox_inches='tight',dpi=240)
          plt.cla()
          print('saved %s' % path,flush=True)
          image_list12.append(imageio.imread(path))          

          writer.flush()  
      if epoch % save_epoch == 0:
          # save the model parameters 
          print("savedmdl_epoch:"+str(epoch))
          g21_path = os.path.join(model_path, 'g21-epoch-%d.pkl' % (epoch))
          d1_path = os.path.join(model_path, 'd1-epoch-%d.pkl' % (epoch))
          torch.save(d1.state_dict(), d1_path)
          torch.save(g21.state_dict(), g21_path)
          g12_path = os.path.join(model_path, 'g12-epoch-%d.pkl' % (epoch))
          d2_path = os.path.join(model_path, 'd2-epoch-%d.pkl' % (epoch))
          torch.save(d2.state_dict(), d2_path)
          torch.save(g12.state_dict(), g12_path)          
      
      if epoch == end_epoch:
        break    
        
      start = time.process_time()



    # load dataset
    train_smpl, _ = train_iter.next()
    noisy_smpl,_ =noisy_iter.next()
    clean = to_var(noisy_smpl[:,:,:,0:256])
    noisy_ref = to_var(noisy_smpl[:,:,:,256:512])    
    clean_ref = to_var(train_smpl[:,:,:,0:256])
    noisy = to_var(train_smpl[:,:,:,256:512])
    ##########3
    comb=torch.cat([clean_ref,noisy,noisy_ref,clean],dim=0)
    t2=transforms.RandomRotation([-20,20])(transforms.RandomHorizontalFlip(p=0.5)(comb))
    clean_ref=t2[0,:,:,:][None,:,:,:]
    noisy=t2[1,:,:,:][None,:,:,:]
    noisy_ref=t2[2,:,:,:][None,:,:,:]
    clean=t2[3,:,:,:][None,:,:,:]     
 ############
    reset_grad()
    
    #identity loss 34an el sora mtt8yr4 awy 
    fake_clean = g21(noisy)
    fake_noisy = g12(clean)
    
    loss_id_A = criterion_identity(fake_noisy,clean)
    loss_id_B = criterion_identity(fake_clean,noisy)
    loss_identity =  (loss_id_A + loss_id_B)/2.0
    
    #GAN Lossa
    
    d1_fake,_=d1(fake_clean)
    
    valid_fh = to_var(torch.tensor(np.ones(d1_fake.shape))).float()
    fake_fh = to_var(torch.tensor(np.zeros(d1_fake.shape))).float()

    loss_GAN_21 = criterion_GAN(d1_fake,valid_fh) 
    
    d2_fake,_=d2(fake_noisy)
    loss_GAN_12 = criterion_GAN(d2_fake,to_var(torch.tensor(np.ones(d2_fake.shape))).float()) #forr generator should be valid 
    loss_GAN = (loss_GAN_12 +loss_GAN_21)/2.0
    
    #percep_loss_clean=percep_loss(d1,fake_clean,clean)
    #percep_loss_noisy=percep_loss(d2,fake_noisy,noisy)  
    #Cycle loss
    reconst_noisy = g12(fake_clean)
    loss_cycle_2 = criterion_cycle(reconst_noisy,noisy)

    reconst_clean = g21(fake_noisy)
    loss_cycle_1 = criterion_cycle(reconst_clean,clean)
    
    loss_cycle = (loss_cycle_1 + loss_cycle_2)/2.0 
    
    percep_loss_clean=percep_loss(d1,reconst_clean,clean)
    percep_loss_noisy=percep_loss(d2,reconst_noisy,noisy)
    percep_loss_total=(percep_loss_clean+percep_loss_noisy)/2.0     
    loss_G = 0.00156639 * loss_identity + 10.1342*loss_cycle + 1.00892*loss_GAN +0.00131397*percep_loss_total  #######################################
    loss_G.backward()
    g_optimizer.step()
    
    
    #train D1
    reset_grad()
    
    d1_real2,_=d1(clean)
    loss_real = criterion_GAN(d1_real2, to_var(torch.tensor(np.ones(d1_real2.shape))).float())
    
    #fake_clean_ = fake_clean_buff.push_and_pop(fake_clean)
    d1_fake2,_ =d1(fake_clean.detach())
    
    loss_fake = criterion_GAN(d1_fake2, to_var(torch.tensor(np.zeros(d1_fake2.shape))).float())
    
    loss_d1 = (loss_real+loss_fake)/2.0 #######################
    loss_d1.backward()
    d_optimizer.step()
    
    reset_grad()
    
    d2_real2,_=d2(noisy)
    loss_real = criterion_GAN(d2_real2, to_var(torch.tensor(np.ones(d2_real2.shape))).float())
    
    #fake_noisy_=fake_noisy_buff.push_and_pop(fake_noisy)
    d2_fake2,_=d2(fake_noisy.detach())

    loss_fake = criterion_GAN(d2_fake2, to_var(torch.tensor(np.zeros(d2_fake2.shape))).float())
    
    loss_d2 = (loss_real+loss_fake)/2.0############3
    loss_d2.backward()
    d2_optimizer.step()

    # print the log info
    if (step+1) % log_step == 0:
         writer.add_scalar("loss_G/step", loss_G.data, step)
         writer.add_scalar("loss_identity/step", loss_identity.data, step)
         writer.add_scalar("loss_cycle/step", loss_cycle.data, step)
         
         writer.add_scalar("loss_GAN/step", loss_GAN.data, step)
         writer.add_scalar("loss_d1/step", loss_d1.data, step)   
         writer.add_scalar("loss_d2/step", loss_d2.data, step) 
         
         writer.add_scalar("percep_loss_clean/step", percep_loss_clean.data, step)                
         writer.add_scalar("percep_loss_noisy/step", percep_loss_noisy.data, step)

  g12_path = os.path.join(model_path, 'g12-last.pkl')
  g21_path = os.path.join(model_path, 'g21-last.pkl')
  d1_path = os.path.join(model_path, 'd1-last.pkl')
  d2_path = os.path.join(model_path, 'd2-last.pkl')
  torch.save(d1.state_dict(), d1_path)
  torch.save(d2.state_dict(), d2_path)
  torch.save(g12.state_dict(), g12_path)
  torch.save(g21.state_dict(), g21_path)
  
  test_g2(g21,step + 1)
  genOut(g21)
  import subprocess
  list_of_files = subprocess.run(['python', '-m','pytorch_fid','./gen_out_c','./gen_out'], capture_output=True, text=True)  
  print(list_of_files.stdout)    
  g21_path = "./models/g21-best.pkl"
  g21.load_state_dict(torch.load(g21_path))
  genOut(g21)
  import subprocess
  list_of_files = subprocess.run(['python', '-m','pytorch_fid','./gen_out_c','./gen_out'], capture_output=True, text=True)
  print("best")
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

  