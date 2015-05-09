
# coding: utf-8

# In[1]:

import emcee
import pyfits
from pylab import*

datos = pyfits.open('hmi.m_45s.magnetogram.subregion_x1y1.fits')
cols = datos[0].data
x = linspace(0, 397, 398)
y = linspace(0, 198, 199)
z = linspace(0, 206, 207)
x_data=np.genfromtxt('times_delta.csv', delimiter=',')



# In[2]:

#funcion lineal_gauss
def model(c_walk, d_walk, sigma_walk, mu_walk, x_d, y_d):
    n1 = sigma_walk**2/(sqrt(2*pi))
    n2 = 0.5*((x_d-mu_walk)/sigma_walk)**2
    y_m = c_walk +d_walk*x_d + n1*exp(-n2)
    return y_m


# In[3]:

#emcee functions
def lnprior(param):
    c_walk, d_walk, sigma_walk, mu_walk= param
    if -200.0 < c_walk < 200.0 and -1 < d_walk < 1 and 34 < sigma_walk < 46 and 94 < mu_walk < 105:
        return 0.0
    return -inf



# In[4]:

def lnlike(param, x_d, y_d):
    c_walk, d_walk, sigma_walk, mu_walk= param
    y_m = model(c_walk, d_walk, sigma_walk, mu_walk, x_d, y_d)
    chi_cuadrado= (1.0/2.0)*sum((y_d-y_m)**2) #campo_m =y_d
    return -chi_cuadrado



# In[5]:

def lnprob(param, x_d, y_d):
    lp = lnprior(param)
    if not isfinite(lp):
        return -inf
    return lp + lnlike(param, x_d, y_d)


# In[6]:

c_walk0 = np.random.random()*400 -200
d_walk0 = np.random.random()*2 -1
sigma_walk0 = np.random.random()*10 + 35
mu_walk0 = np.random.random()*10.0 + 95.0



# In[7]:

#iteracion de los n pixeles
titulo = 'Ajuste Lineal Gauss'
info = "x,y son las coordenas del pixel analizado en el archivo 'hmi.m_45s.magnetogram.subregion_x1y1.fits'. best_b y best_a son los parametros de maximo likelihood para el modelo lineal"
heading = 'x,y,c,d,sigma,mu' #modificar cada parametro
f = open('Modeloemcee_Gauss.csv','w')
f.write(titulo+'\n')
f.write(info+'\n')
f.write(heading+'\n')

#coordenadas del pixel seleccionado

#TOCA CAMBIAR A IN RANGE 100

x = [2*i+1 for i in range(100)]
y = [ i +2 for i in range(100)]


#Running emcee
ndim = 4 #numero de walkers
nwalkers = 8
nsteps = 20000

n=len(x)

def correr_emcee():
    for i in range (n):
        y_data=cols[:,x[i],y[i]]
    
        first_guess = [c_walk0, d_walk0, sigma_walk0, mu_walk0]
        pos0 = [first_guess+ 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x_data, y_data), threads=1)
        sampler.run_mcmc(pos0, nsteps, rstate0=np.random.get_state())
        s_fc= sampler.flatchain
        
        c,bins,info = hist(s_fc[:,0], 207)
        id_max = argmax(c)
        c_walk = mean([bins[id_max],bins[id_max+1]])
        
        d,bins,info = hist(s_fc[:,1], 207)
        id_max = argmax(d)
        d_walk = mean([bins[id_max],bins[id_max+1]])
        
        sig,bins,info = hist(s_fc[:,2], 207)
        id_max = argmax(sig)
        sig_walk = mean([bins[id_max],bins[id_max+1]])
        
        mu,bins,info = hist(s_fc[:,3], 207)
        id_max = argmax(mu)
        mu_walk = mean([bins[id_max],bins[id_max+1]])
        
        
        aja=model(c_walk,d_walk,sig_walk,mu_walk,x_data, y_data)
        plot(x_data,aja)
        scatter(x_data,y_data)
        show()
        
        f.write(str(x[i]) + ',' + str(y[i]) + ',' + '%.7f'%c_walk + ',' + '%.7f'%d_walk + ',' + '%.7f'%sig_walk + ',' + '%.7f'%mu_walk + '\n')
    f.close()   
    return c_walk, d_walk, sig_walk, mu_walk


# In[8]:

correr_emcee()


# In[59]:

savetxt('sampler_flatchainGauss.dat', s_fc, delimiter=',')
print("Mean acceptance fraction: {0:.3f}".format(mean(sampler.acceptance_fraction)))


# In[38]:

#samples = sampler.chain[:, 207:, :].reshape((-1, ndim))
#samples = sampler.flatchain
#print samples
#print shape(samples)
#import triangle
#fig = triangle.corner(samples, labels=["$c$", "$d$","$sigma$", "$mu$" ],
#                      truths=[c_walk0, d_walk0, sigma_walk0, mu_walk0])
#fig.savefig("triangle_Gauss.png")


# In[ ]:



