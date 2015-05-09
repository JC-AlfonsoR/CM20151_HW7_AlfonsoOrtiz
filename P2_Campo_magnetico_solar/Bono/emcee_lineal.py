
# coding: utf-8

# In[5]:

import emcee
import pyfits
from pylab import*
import scipy.optimize as op

datos = pyfits.open('hmi.m_45s.magnetogram.subregion_x1y1.fits')
cols = datos[0].data
x = linspace(0, 397, 398)
y = linspace(0, 198, 199)
z = linspace(0, 206, 207)
x_data=np.genfromtxt('times_delta.csv', delimiter=',')



# In[6]:

def model(a_walk, b_walk, x_d, y_d): #dejar x y y?
    y_m = x_d*b_walk + a_walk
    return y_m


# In[7]:

#emcee functions
def lnprior(param):
    a_walk, b_walk= param
    if -200.0 < a_walk < 200.0 and -1 < b_walk < 1:
        return 0.0
    return -inf


# In[8]:

def lnlike(param, x_d, y_d):
    a_walk, b_walk= param
    y_m = model(a_walk, b_walk, x_d, y_d)
    chi_cuadrado= (1.0/2.0)*sum((y_d-y_m)**2) #campo_m =y_d
    return -chi_cuadrado


# In[9]:

def lnprob(param, x_d, y_d):
    lp = lnprior(param)
    if not isfinite(lp):
        return -inf
    return lp + lnlike(param, x_d, y_d)


# In[10]:

#iteracion de los n pixeles
titulo = 'Ajuste Lineal Gauss'
info = "x,y son las coordenas del pixel analizado en el archivo 'hmi.m_45s.magnetogram.subregion_x1y1.fits' modelo lineal"
heading = 'x,y,a,b' #modificar cada parametro
f = open('Modeloemcee_lineal.csv','w')
f.write(titulo+'\n')
f.write(info+'\n')
f.write(heading+'\n')

#Running emcee

#TOCA CAMBIAR A IN RANGE 100

x = [2*i+1 for i in range(100)]
y = [ i +2 for i in range(100)]


ndim = 2 #numero de walkers
nwalkers = 4
nsteps = 20000

n=len(x)
a_walk0 = np.random.random()*400 -200
b_walk0 = np.random.random()*2 -1

def correr_emcee():
    for i in range (n):
        
        y_data=cols[:,x[i],y[i]]
        
      
        nll = lambda *args: -lnlike(*args)
        result = op.minimize(nll, [a_walk0, b_walk0], args=(x_data, y_data))
        a_ml, b_ml = result["x"]
        
        pos0 = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x_data, y_data), threads=1)
        sampler.run_mcmc(pos0, nsteps, rstate0=np.random.get_state())
        
        s_fc= sampler.flatchain
        
             
        a,bins,info = hist(s_fc[:,0], 207)
       
        id_max = argmax(a)
        a_walk = mean([bins[id_max],bins[id_max+1]])
        
        b,bins,info = hist(s_fc[:,1], 207)
        
        id_max = argmax(b)
        b_walk = mean([bins[id_max],bins[id_max+1]])
        
        f.write(str(x[i]) + ',' + str(y[i]) + ',' + '%.7f'%a_walk + ',' + '%.7f'%b_walk + '\n')

        aja=model(a_walk,b_walk,x_data, y_data)
        plot(x_data,aja)
        scatter(x_data,y_data)
        show()
    
    f.close()
        
    return a_walk, b_walk


# In[1]:


correr_emcee()


# In[ ]:

#samples = sampler.chain[:, 207:, :].reshape((-1, ndim))
#samples = sampler.flatchain
#print samples
#print shape(samples)
#import triangle
#fig = triangle.corner(samples, labels=["$a$", "$b$"],
#                      truths=[a_walk0, b_walk0])
#fig.savefig("triangle.png")


# In[48]:

#savetxt('sampler_flatchainLineal.dat', s_fc, delimiter=',')
#print("Mean acceptance fraction: {0:.3f}".format(mean(sampler.acceptance_fraction)))

