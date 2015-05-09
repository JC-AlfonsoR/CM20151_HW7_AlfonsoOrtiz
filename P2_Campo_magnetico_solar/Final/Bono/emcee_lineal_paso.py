
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

def model(f_walk,g_walk,h_walk,n_walk,t0_walk,x_d, y_d):
    y_m = f_walk + g_walk*x_d + h_walk*(1 + 2/pi*arctan(n_walk*(x_d-t0_walk)))
    return y_m


# In[3]:

#emcee functions
def lnprior(param):
    f_walk,g_walk,h_walk,n_walk,t0_walk= param
    if -200.0 < f_walk < 200.0 and -1 < g_walk < 1 and -100.0 < h_walk < 100.0 and 0 < n_walk < 100 and 80 < t0_walk < 350:
        return 0.0
    return -inf


# In[4]:

def lnlike(param, x_d, y_d):
    f_walk,g_walk,h_walk,n_walk,t0_walk= param
    y_m = model(f_walk,g_walk,h_walk,n_walk,t0_walk, x_d, y_d)
    chi_cuadrado= (1.0/2.0)*sum((y_d-y_m)**2) 
    return -chi_cuadrado


# In[5]:

def lnprob(param, x_d, y_d):
    lp = lnprior(param)
    if not isfinite(lp):
        return -inf
    return lp + lnlike(param, x_d, y_d)


# In[6]:

#iteracion de los n pixeles
titulo = 'Ajuste Lineal paso'
info = "x,y son las coordenas del pixel analizado en el archivo 'hmi.m_45s.magnetogram.subregion_x1y1.fits' modelo lineal"
heading = 'x,y,f,g,h,n,t0' #modificar cada parametro
f = open('Modeloemcee_linealPaso.csv','w')
f.write(titulo+'\n')
f.write(info+'\n')
f.write(heading+'\n')

#Running emcee

#TOCA CAMBIAR A IN RANGE 100 y nstetps a 2000

x = [2*i+1 for i in range(100)]
y = [ i +2 for i in range(100)]


ndim = 5 #numero de walkers
nwalkers = 10
nsteps = 20000


n=len(x)

f_walk0 = np.random.random()*400 - 200
g_walk0 = np.random.random()*2.0 - 1.0
h_walk0 = np.random.random()*200 - 100
n_walk0 = np.random.random()*100
t0_walk0 = np.random.random()*270 + 80 # t0 entre 80 y 350

def correr_emcee():
    n=len(x)
    for i in range (n):
        
        y_data=cols[:,x[i],y[i]]
        #Running emcee
        ndim = 5 #numero de walkers
        nwalkers = 10
        nsteps = 2000

        first_guess = [f_walk0,g_walk0,h_walk0,n_walk0,t0_walk0]
        pos0 = [first_guess+ 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x_data, y_data), threads=1)
        sampler.run_mcmc(pos0, nsteps, rstate0=np.random.get_state())

        s_fc= sampler.flatchain
        
        f,bins,info = hist(s_fc[:,0], 207)
        id_max = argmax(f)
        f_walk = mean([bins[id_max],bins[id_max+1]])

        g,bins,info = hist(s_fc[:,1], 207)
        id_max = argmax(g)
        g_walk = mean([bins[id_max],bins[id_max+1]])

        h,bins,info = hist(s_fc[:,2], 207)
        id_max = argmax(h)
        h_walk = mean([bins[id_max],bins[id_max+1]])

        n,bins,info = hist(s_fc[:,3], 207)
        id_max = argmax(n)
        n_walk = mean([bins[id_max],bins[id_max+1]])

        t0,bins,info = hist(s_fc[:,4], 207)
        id_max = argmax(t0)
        t0_walk = mean([bins[id_max],bins[id_max+1]])

        aja=model(f_walk,g_walk,h_walk,n_walk,t0_walk,x_data, y_data)
        plot(x_data,aja)
        scatter(x_data,y_data)
        show()
        
        f.write(str(x[i]) + ',' + str(y[i]) + ',' + '%.7f'%f_walk + ',' + '%.7f'%g_walk + ',' + '%.7f'%h_walk + ',' + '%.7f'%n_walk + ',' + '%.7f'%t0_walk + '\n')
    f.close()
    return f_walk, g_walk, h_walk, n_walk, t0_walk
    


# In[7]:

correr_emcee()


# In[84]:

#samples = sampler.chain[:, 207:, :].reshape((-1, ndim))
#samples = sampler.flatchain
#print samples
#print shape(samples)
#import triangle
#fig = triangle.corner(samples, labels=["$f$", "$g$", "$h$", "$n$", "$t0$"],
#                      truths=[f_walk0,g_walk0,h_walk0,n_walk0,t0_walk0,x_data, y_data])
#fig.savefig("trianglePaso.png")


# In[73]:

#savetxt('sampler_flatchain_linealpaso.dat', s_fc, delimiter=',')
#print("Mean acceptance fraction: {0:.3f}".format(mean(sampler.acceptance_fraction)))

