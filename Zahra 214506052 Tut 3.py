# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:28:37 2017

@author: Zahra
"""

import numpy as np

import matplotlib.pyplot as plt

class NbodySolver:

    def __init__(self, N=100, soften=0.1, m=1.0, dt=0.01):

        self.options = {}  
        self.options['dt'] = dt
        self.options['G'] = 1.0             #dictionary that contains G
        self.options['soften'] = soften
        self.options['N'] = N               #Dictionary that contains number of particles
        self.x = np.random.randn(self.options['N']) #Problem 2 A
        self.y = np.random.randn(self.options['N'])
        self.m = np.ones(self.options['N']) * (m / self.options['N'])
        self.vx = np.zeros(self.options['N'])
        self.vy = np.zeros(self.options['N'])
        
        def __GravPot__(self):              #For problem 1

            potential = np.zeros(self.options['N'])

            for i in range(0, self.options['N']):

                for j in range(0, self.options['N']):

                    if (i != j):

                        xarr = self.x[i] - self.x[j]

                        yarr = self.y[i] - self.y[j]

                        radius = np.sqrt(xarr ** 2 + yarr ** 2)

                        potential[i] = potential[i] + self.m[i] * self.m[j] * self.options['G'] * 1.0 / radius

            return potential

    def force(self):    #Problem 2B

        self.fx = np.zeros(self.options['N'])

        self.fy = np.zeros(self.options['N'])

        potential = np.zeros(self.options['N'])

        for i in range(0, self.options['N'] - 1):

            for j in range(i + 1, self.options['N']):

                xarr = self.x[i] - self.x[j]

                yarr = self.y[i] - self.y[j]

                radius2 = xarr ** 2 + yarr ** 2

                softened = self.options['soften'] ** 2



                if radius2 < softened:

                    radius2 = softened



                radius = np.sqrt(radius2)

                Rnew = radius * radius2             #New radius


                self.fx[i] = self.fx[i] - xarr * (1.0 / Rnew) * self.m[j]

                self.fy[i] = self.fy[i] - yarr * (1.0 / Rnew) * self.m[j]

                self.fx[j] = self.fx[j] + xarr * (1.0 / Rnew) * self.m[i]

                self.fy[j] = self.fy[j] + yarr * (1.0 / Rnew) * self.m[i]



                potential[i] = potential[i] + self.m[i] * self.m[j] * self.options['G'] * 1.0 / radius

        return potential.sum()


    def Update(self, dt=0.01):          #Problem 2C

        self.x = self.x + self.vx * self.options['dt']

        self.y = self.y + self.vy * self.options['dt']

        potential = self.force()

        self.vx = self.vx + self.fx * self.options['dt']

        self.vy = self.vy + self.fy * self.options['dt']

        return potential.sum()
        
nclass = NbodySolver()

print 'energy is ', nclass.force()


potentialvar = np.zeros(100)
kineticvar = np.zeros(100)
dx = 0

while (dx < 100):

    finalPotential = nclass.Update(0.05)

    kineticenergy = np.sum(nclass.m * (nclass.vx ** 2 + nclass.vy ** 2))

    print 'Total energy is ', [finalPotential, kineticenergy, kineticenergy - 2.0 * finalPotential]

    plt.plot(nclass.x, nclass.y, 'b*')          #Problem 2D

    plt.show()

    potentialvar[dx] = finalPotential

    kineticvar[dx] = kineticenergy

    dx += 1
    
#Problem 3
    
import numpy as np

import matplotlib.pyplot as plt

N = 500

x = np.linspace(0,2*np.pi,N)

y = np.sin(x)
z=np.cos(x)

data = y  + np.random.randn(x.size)

order = 10

A = np.zeros([x.size,order])

A[:,0] = 1.0

for i in range(1,order):

    A[:,i]=A[:,i-1]*x

A = np.matrix(A)

d = np.matrix(data).transpose()

lhs = A.transpose()*A

rhs = A.transpose()*d

fitp = np.linalg.inv(lhs)*rhs

pred = A*fitp

plt.plot(x,y)
plt.plot(x,z)
plt.plot(x,data,'*')
plt.plot(x,pred,'r')

plt.show()

#Problem 4

def sim_lor(t, a=1, b=1, c=0):

    dat = a/(b+(t-c)**2)

    dat += np.random.randn(t.size)

    return dat

class Lorentz:

    def __init__(self, t, a=1, b=0.5, c=0, offset=0):

        self.t = t

        self.y = sim_lor(t, a, b, c) + offset

        self.error = np.ones(t.size)

        self.a = a

        self.b = b

        self.c = c



    def get_chis(self, vec):

        a = vec[0]

        b = vec[1]

        c = vec[2]

        offset = vec[3]

        pred = offset + a / (b + (self.t - c) ** 2)

        chisq = np.sum((self.y - pred) ** 2 / self.error ** 2)

        return chisq

def get_trial_offset(sigs):

    return sigs * np.random.randn(sigs.size)

def run_mcmc(data, start_pos, nstep, scale=None):

    nparam = start_pos.size

    params = np.zeros([nstep, nparam + 1])

    params[0, 0:-1] = start_pos

    cur_chisq = data.get_chisq(start_pos)

    cur_pos = start_pos.copy()

    if scale == None:

        scale = np.ones(nparam)

    for i in range(1, nstep):

        new_pos = cur_pos + get_trial_offset(scale)

        new_chisq = data.get_chisq(new_pos)

        if new_chisq < cur_chisq:

            accept = True

        else:

            delt = new_chisq - cur_chisq

            prob = np.exp(-0.5 * delt)

            if np.random.rand() < prob:

                accept = True

            else:

                accept = False

        if accept:

            cur_pos = new_pos

            cur_chisq = new_chisq

        params[i, 0:-1] = cur_pos

        params[i, -1] = cur_chisq

    return params

t = np.arange(-5, 5, 0.01)

data = Lorentz(t, a=5)

guess = np.array([1.2, 0.3, 0.3, -0.2])
scale = np.array([0.1, 0.1, 0.1, 0.1])
nstep = 100000
chain = run_mcmc(data, guess, nstep, scale)
nn = int(np.round(0.2 * nstep))
chain = chain[nn:, :]  # removing the first few bad values
true_param = np.array([data.a, data.b, data.c])

for i in range(0, true_param.size):

    val = np.mean(chain[:, i])
    scat = np.std(chain[:, i])
    print(true_param[i], val, scat)