import torch
import torch.nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
mb_size = 32
z_dim = 10
X_dim = mnist.train.images.shape[1]
print(X_dim)
y_dim = mnist.train.labels.shape[1]
print(y_dim)
h_dim = 128
cnt = 0
d_step = 3
lr = 1e-3
losses=[]

class Generator(torch.nn.Module):
    def __init__(self,z_dim,h_dim,X_dim):
        super(Generator,self).__init__()
        self.l1 = torch.nn.Linear(z_dim, h_dim)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(h_dim, X_dim)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self,z_dim,h_dim,X_dim):
        super(Discriminator,self).__init__()
        self.relu = torch.nn.ReLU()
        self.l1 = torch.nn.Linear(X_dim, h_dim)
        self.l2 = torch.nn.Linear(h_dim, 1)
    def forward(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

G = Generator(z_dim,h_dim,X_dim)
D = Discriminator(z_dim,h_dim,X_dim)

def reset_grad():
    G.zero_grad()
    D.zero_grad()


G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)


for it in range(1000000):
    for _ in range(d_step):
        # Sample data
        z = Variable(torch.randn(mb_size, z_dim))
        X, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X))

        # Dicriminator
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss = 0.5 * (torch.mean((D_real - 1)**2) + torch.mean(D_fake**2))

        D_loss.backward()
        D_solver.step()
        reset_grad()

    # Generator
    z = Variable(torch.randn(mb_size, z_dim))

    G_sample = G(z)
    D_fake = D(G_sample)

    G_loss = 0.5 * torch.mean((D_fake - 1)**2)

    G_loss.backward()
    G_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.data[0], G_loss.data[0]))

        samples = G(z).data.numpy()[:16]
        #losses
        losses.append((D_loss.data[0],G_loss.data[0]))
        #save model
        if not os.path.exists('models/'):
            os.makedirs('models/')
        filename="models/"+str(cnt)+".pt"
        torch.save(D.state_dict(),filename)
        torch.save(G.state_dict(),filename)

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)


        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
losses=np.asarray(losses)
f=open("losses.npy","wb")
f1=np.save(f,losses)
f.close()
