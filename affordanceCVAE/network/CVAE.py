import torch
import torch.nn as nn



class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, condition_size=1024):
        super().__init__()

        if conditional:
            assert condition_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, condition_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, condition_size)

    def forward(self, x, c=None):
        # x: [B, 58]
        # if x.dim() > 2:
        #     x = x.view(-1, 58)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size], device=c.device)
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        #print('encoder', self.MLP)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1) # [B, 1024+61]
        #print('x size before MLP {}'.format(x.size()))
        x = self.MLP(x)
        #print('x size after MLP {}'.format(x.size()))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        #print('mean size {}, log_var size {}'.format(means.size(), log_vars.size()))
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        #print('decoder', self.MLP)

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)
            #print('z size {}'.format(z.size()))

        x = self.MLP(z)

        return x

if __name__ == '__main__':
    model = VAE(
        encoder_layer_sizes=[61, 512, 256],
        latent_size=64,
        decoder_layer_sizes=[1024, 256, 61],
        condition_size=1024)

    input = torch.randn((5, 61))
    condition = torch.randn((5, 1024))
    print(input.size(), condition.size())
    print('params {}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    # model.train()
    # recon, _, _, _ = model(input, condition)
    model.eval()
    recon = model.inference(input.size(0), condition)
    print('recon size {}'.format(recon.size()))
