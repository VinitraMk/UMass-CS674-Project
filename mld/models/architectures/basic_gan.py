import torch
import torch.nn as nn
import pytorch_lightning as pl

class Generator(nn.Module):
  '''
  Generator class in a CGAN. Accepts a noise tensor (latent dim 100)
  and a label tensor as input as outputs another tensor of size 784.
  Objective is to generate an output tensor that is indistinguishable 
  from the real MNIST digits.
  '''

  def __init__(self, latent_in, text_in, latent_out):
    super().__init__()
    self.layer1 = nn.Sequential(nn.Linear(in_features=latent_in+text_in, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=512, out_features=latent_out),
                                )

  def forward(self, z, text_emb):
    # x is a tensor of size (batch_size, 110)
    # reshapeing text_emb
    text_emb = text_emb.reshape(text_emb.shape[0],-1)
    print(text_emb.shape, z.shape)
    # print(z.shape)
    # print(text_emb.shape)
    x = torch.cat([z, text_emb], dim=-1)    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.output(x)
    print('out shape gen', x.shape)
    return x


class Discriminator(nn.Module):
  '''
  Discriminator class in a CGAN. Accepts a tensor of size 784 and
  a label tensor as input and outputs a tensor of size 1,
  with the predicted class probabilities (generated or real data)
  '''

  def __init__(self, latent_in, text_in, latent_out):
    super().__init__()
    self.layer1 = nn.Sequential(nn.Linear(in_features=latent_out+text_in, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=256, out_features=1),
                                nn.Sigmoid())
    
  def forward(self, x, text_emb):
    # pass the labels into a embedding layer
    # labels_embedding = self.embedding(y)
    # concat the embedded labels and the input tensor
    # x is a tensor of size (batch_size, 794)
    text_emb = text_emb.reshape(text_emb.shape[0],-1)
    x = torch.cat([x, text_emb], dim=-1)    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x


class CGAN(pl.LightningModule):

  def __init__(self, latent_in_dim, text_emb_dim, latent_out_dim):
    super().__init__()
    self.latent_in_dim = latent_in_dim
    self.text_emb_dim = text_emb_dim
    self.latent_out_dim = latent_out_dim
    self.generator = Generator(latent_in_dim, text_emb_dim, latent_out_dim)
    self.discriminator = Discriminator(latent_in_dim, text_emb_dim, latent_out_dim)
    self.BCE_loss = nn.BCELoss()

  def forward(self, z, text_emb):
    """
    Generates an image using the generator
    given input noise z and labels y
    """
    return self.generator(z, text_emb)

  def generator_step(self, z, text_emb):
    """
    Training step for generator
    1. Sample random noise and labels
    2. Pass noise and labels to generator to
       generate images
    3. Classify generated images using
       the discriminator
    4. Backprop loss
    """

    # Generate images
    fake_latent = self(z, text_emb)

    # Classify generated image using the discriminator
    fake_pred = torch.squeeze(self.discriminator(fake_latent, text_emb))

    # Backprop loss. We want to maximize the discriminator's
    # loss, which is equivalent to minimizing the loss with the true
    # labels flipped (i.e. y_true=1 for fake images). We do this
    # as PyTorch can only minimize a function instead of maximizing
    g_loss = self.BCE_loss(fake_pred, torch.ones_like(fake_pred))

    return g_loss

  def discriminator_step(self, z, x, text_emb):
    """
    Training step for discriminator
    1. Get actual images and labels
    2. Predict probabilities of actual images and get BCE loss
    3. Get fake images from generator
    4. Predict probabilities of fake images and get BCE loss
    5. Combine loss from both and backprop
    """
    
    # Real images
    x = x.reshape(z.shape[0], -1)
    real_pred = torch.squeeze(self.discriminator(x, text_emb))
    real_loss = self.BCE_loss(real_pred, torch.ones_like(real_pred))


    fake_latent = self(z, text_emb).detach() 
    fake_pred = torch.squeeze(self.discriminator(fake_latent, text_emb))
    fake_loss = self.BCE_loss(fake_pred, torch.zeros_like(fake_pred))


    d_loss = (real_loss + fake_loss) / 2
    return d_loss

    
  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.002)
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.002)
    return [g_optimizer, d_optimizer], []


# if __name__ == "__main__":
#   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#   mnist_transforms = transforms.Compose([transforms.ToTensor(),
#                                       transforms.Normalize(mean=[0.5], std=[0.5]),
#                                       transforms.Lambda(lambda x: x.view(-1, 784)),
#                                       transforms.Lambda(lambda x: torch.squeeze(x))
#                                       ])

#   data = datasets.MNIST(root='../data/MNIST', download=True, transform=mnist_transforms)

#   mnist_dataloader = DataLoader(data, batch_size=128, shuffle=True, num_workers=0) 

#   model = CGAN()

#   trainer = pl.Trainer(max_epochs=100, gpus=1 if torch.cuda.is_available() else 0, progress_bar_refresh_rate=50)
#   trainer.fit(model, mnist_dataloader)
  
  