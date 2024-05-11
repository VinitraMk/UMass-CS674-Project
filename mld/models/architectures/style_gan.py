import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8]
#factors = [1, 1, 1, 1, 1, 1, 1, 1, 1]

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #print('pn', x.shape)
        out = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        #print('pn out', out.shape)
        return out

class WSLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        #print('ws line', x.shape)
        return self.linear(x * self.scale) + self.bias

class WSConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0
    ):
        super(WSConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv1d(in_channels, out_channels)
        self.conv2 = WSConv1d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x

class MappingNetwork(nn.Module):

    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
        )

    def forward(self, x):
        return self.mapping(x)


class AdaIN(nn.Module):

    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)

    def forward(self, x, w):
        #print('adain========>', x.shape, w.shape)
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2)
        style_bias = self.style_bias(w).unsqueeze(2)
        #print('st', style_scale.shape, x.shape, style_bias.shape)
        return style_scale * x + style_bias

class InjectNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        #print('inoise', x.shape, self.weight.shape)
        noise = torch.randn((x.shape[0], 1, x.shape[2]), device=x.device)
        out = x + self.weight * noise
        #print('out', out.shape)
        return out


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv1d(in_channels, out_channels)
        self.conv2 = WSConv1d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)

    def forward(self, x, w):
        #print('gb inp', x.shape, w.shape, self.conv1(x).shape)
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x




class Generator(nn.Module):
  '''
  Generator class in a CGAN. Accepts a noise tensor (latent dim 100)
  and a label tensor as input as outputs another tensor of size 784.
  Objective is to generate an output tensor that is indistinguishable 
  from the real MNIST digits.
  '''

  def __init__(self, latent_in, text_in, latent_out, in_channels = 128, xyz_channels = 3):
    super().__init__()
    #in_channels = latent_in + text_in
    print('args', latent_in, text_in, latent_out, in_channels)
    self.starting_constant = nn.Parameter(torch.ones((1, in_channels, 21)))
    self.map = MappingNetwork(text_in + latent_in, latent_out)
    self.initial_adain1 = AdaIN(in_channels, latent_out)
    self.initial_adain2 = AdaIN(in_channels, latent_out)
    self.initial_noise1 = InjectNoise(in_channels)
    self.initial_noise2 = InjectNoise(in_channels)
    self.initial_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride = 1)
    self.leaky = nn.LeakyReLU(0.2, inplace = True)

    self.initial_xyz = WSConv1d(in_channels, xyz_channels, kernel_size = 1, stride = 1)
    self.prog_blocks, self.xyz_layers = (nn.ModuleList([]), nn.ModuleList([self.initial_xyz]))

    for i in range(len(factors) - 1):
        conv_in_c = int(in_channels * factors[i])
        conv_out_c = int(in_channels * factors[i + 1])
        #print('prog setup', conv_in_c, conv_out_c, in_channels, factors[i+1], factors[i])
        self.prog_blocks.append(GenBlock(conv_in_c, conv_out_c, latent_out))
        self.xyz_layers.append(
            WSConv1d(conv_out_c, xyz_channels, kernel_size = 1, stride = 1, padding = 0)
        )

    '''
    self.layer1 = nn.Sequential(nn.Linear(in_features=latent_in+text_in, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=512, out_features=latent_out),
                                )
    '''

  def fade_in(self, alpha, upscaled, generated):
    return torch.tanh(alpha * generated * (1 - alpha) * upscaled)

  def forward(self, z, text_emb, alpha = 1.0, steps = 6):

    # x is a tensor of size (batch_size, 110)
    # reshapeing text_emb
    print('z, txt', z.shape, text_emb.shape)
    text_emb = text_emb.reshape(text_emb.shape[0],-1)
    x = torch.cat([z, text_emb], dim=-1)
    #print('x', x.shape, z.shape)
    noise = torch.randn(x.shape[0], x.shape[1]).to(torch.device("cuda"))
    w = self.map(noise)
    #print('w', w.shape)
    x = self.initial_adain1(self.initial_noise1(self.starting_constant), w)
    x = self.initial_conv(x)
    #print('x out', x.shape)
    out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)
    #print('out', out.shape)

    if steps == 0:
        return self.initial_xyz(x)
    
    for step in range(steps):
        upscaled = F.interpolate(out, scale_factor = 2, mode = 'linear')
        #print('step pg', step, upscaled.shape, w.shape)
        out = self.prog_blocks[step](upscaled, w)
    

    final_upscaled = self.xyz_layers[steps - 1](upscaled)
    final_out = self.xyz_layers[steps](out)
    final_out = self.fade_in(alpha, final_upscaled, final_out)
    print('final out gen', final_out.shape)
    return final_out
    '''
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.output(x)
    return x
    '''



class Discriminator(nn.Module):
  '''
  Discriminator class in a CGAN. Accepts a tensor of size 784 and
  a label tensor as input and outputs a tensor of size 1,
  with the predicted class probabilities (generated or real data)
  '''

  def __init__(self, in_channels = 128, xyz_channels = 3):
    super(Discriminator, self).__init__()
    self.prog_blocks, self.xyz_layers = nn.ModuleList([]), nn.ModuleList([])
    self.leaky = nn.LeakyReLU(0.2)

    for i in range(len(factors) - 1, 0, -1):
        conv_in = int(in_channels * factors[i])
        conv_out = int(in_channels * factors[i - 1])
        self.prog_blocks.append(ConvBlock(conv_in, conv_out))
        self.xyz_layers.append(
            WSConv1d(xyz_channels, conv_in, kernel_size=1, stride=1, padding=0)
        )
    
    self.initial_xyz = WSConv1d(xyz_channels, in_channels, kernel_size = 1, stride = 1, padding = 0)
    self.xyz_layers.append(self.initial_xyz)
    self.avg_pool = nn.AvgPool1d(kernel_size = 2, stride = 2)
    self.final_block = nn.Sequential(
        WSConv1d(in_channels + 1, in_channels, kernel_size = 1),
        nn.LeakyReLU(0.2),
        WSConv1d(in_channels, in_channels, kernel_size = 1, stride = 1),
        nn.LeakyReLU(0.2),
        WSConv1d(in_channels, 1, kernel_size = 1, stride = 1)
    )
    self.final_prob_layer = WSLinear(21, 1)
    '''
    self.layer1 = nn.Sequential(nn.Linear(in_features=latent_out+text_in, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=256, out_features=1),
                                nn.Sigmoid())
    '''

  def fade_in(self, alpha, downscaled, out):
    return alpha * out + (1 - alpha) * downscaled

  def minibatch_std(self, x):
    batch_statistics = (
        torch.std(x, dim = 0).mean().repeat(x.shape[0], 1, x.shape[2])
    )
    return torch.cat([x, batch_statistics], 1)

  def forward(self, x, alpha, steps):
    # pass the labels into a embedding layer
    # labels_embedding = self.embedding(y)
    # concat the embedded labels and the input tensor
    # x is a tensor of size (batch_size, 794)
    '''
    text_emb = text_emb.reshape(text_emb.shape[0],-1)
    x = torch.cat([x, text_emb], dim=-1)    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x
    '''
    print('disc inp shape', x.shape)
    cur_step = len(self.prog_blocks) - steps
    out = self.leaky(self.xyz_layers[cur_step](x))
    print('after leaky', out.shape)

    if steps == 0:
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

    downscaled = self.leaky(self.xyz_layers[cur_step + 1](self.avg_pool(x)))
    out = self.avg_pool(self.prog_blocks[cur_step](out))

    out = self.fade_in(alpha, downscaled, out)

    for step in range(cur_step + 1, len(self.prog_blocks)):
        #print('prog', out.shape)
        out = self.prog_blocks[step](out)
        #print('prg block', out.shape)
        out = self.avg_pool(out)
        #print('avg pool block', out.shape)

    out = self.minibatch_std(out)
    #print('out b4', out.shape)
    out = self.final_block(out).view(out.shape[0], -1)
    #print('final block', out.shape)
    out = self.final_prob_layer(out)
    print('final disc out', out.shape)
    return out

class CGAN(pl.LightningModule):

  def __init__(self, latent_in_dim, text_emb_dim, latent_out_dim):
    super().__init__()
    self.latent_in_dim = latent_in_dim
    self.text_emb_dim = text_emb_dim
    self.latent_out_dim = latent_out_dim
    self.generator = Generator(latent_in_dim, text_emb_dim, latent_out_dim)
    self.discriminator = Discriminator()
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
    print('gs step', z.shape, text_emb.shape)
    fake_latent = self(z, text_emb)

    # Classify generated image using the discriminator
    print('fake latent shape', fake_latent.shape)
    fake_pred = self.discriminator(fake_latent, 1.0, 6)

    # Backprop loss. We want to maximize the discriminator's
    # loss, which is equivalent to minimizing the loss with the true
    # labels flipped (i.e. y_true=1 for fake images). We do this
    # as PyTorch can only minimize a function instead of maximizing
    g_loss = self.BCE_loss(fake_pred, torch.ones_like(fake_pred))
    print('gloss', g_loss.shape, g_loss)

    return g_loss

  def discriminator_step(self, z_fake, z, text_emb):
    """
    Training step for discriminator
    1. Get actual images and labels
    2. Predict probabilities of actual images and get BCE loss
    3. Get fake images from generator
    4. Predict probabilities of fake images and get BCE loss
    5. Combine loss from both and backprop
    """
    
    # Real images
    print('======================')
    print('b4 reshape real shape', z_fake.shape, z.shape, text_emb.shape)
    #x = x.reshape(z_fake.shape[0], -1)
    #print('after reshape real shape', x.shape)
    #real_pred = torch.squeeze(self.discriminator(x, 1.0, 6))
    fake_pred = self.discriminator(z_fake, 1.0, 6)
    fake_loss = self.BCE_loss(fake_pred, torch.ones_like(fake_pred))

    print('real latent b4 self', z.shape, text_emb.shape)
    real_latent = self(z.squeeze(), text_emb).detach() 
    print('generated latent', real_latent.shape)
    #fake_pred = torch.squeeze(self.discriminator(fake_latent, 1.0, 6))
    real_pred = self.discriminator(real_latent, 1.0, 6)
    print('fake pred', real_pred.shape)
    real_loss = self.BCE_loss(real_pred, torch.zeros_like(real_pred))


    d_loss = (real_loss + fake_loss) / 2
    print('dloss', d_loss.shape, d_loss)
    return d_loss

    
  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
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
  
  