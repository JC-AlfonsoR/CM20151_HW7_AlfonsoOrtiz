
# coding: utf-8

# In[1]:

from pylab import*
import pyfits
from scipy.optimize import curve_fit

datos = pyfits.open('hmi.m_45s.magnetogram.subregion_x1y1.fits')
cols = datos[0].data
#arreglo para el tiempo
tiempo=np.genfromtxt('times_delta.csv', delimiter=',')
#coordenadas del pixel seleccionado


x = [2*i+1 for i in range(100)]
y = [ i +2 for i in range(100)]



# ###Lineal-paso
# $$B(t)=f+gt+h\left(1+\frac{2}{\pi}tan^{-1}\left(n(t-t_0)\right)\right)$$

# In[2]:

def paso(t,f,g,h,n,t0):
    return f + g*t + h*(1 + 2/pi*arctan(n*(t-t0)))
#funcion de likelihood lineal, sin exp
def likelihood(y_obs,y_m):
    chi_cuadrado= (1.0/2.0)*sum((y_obs-y_m)**2) # Se esta asumiendo que los sigmas = 1
    return -chi_cuadrado


# In[3]:

def mcmc_lineal_paso(tiempo, campo):

    f_walk = empty((0))
    g_walk = empty((0))
    h_walk = empty((0))
    n_walk = empty((0))
    t0_walk = empty((0))
    like_walk = empty((0))

    mm = mean(campo)
    dd = std(campo)
    nn = (campo[-1]-campo[0])/(tiempo[-1]-tiempo[0])
    mm_t = mean(tiempo)


    f_walk = append(f_walk,np.random.random()*200-100)
    g_walk = append(g_walk,np.random.random()*2.0 - 1.0)
    h_walk = append(h_walk,np.random.random()*150.0 - 75.0)
    n_walk = append(n_walk,np.random.random()/100)
    t0_walk = append(t0_walk,np.random.random()*270+80) # t0 entre 80 y 350

    y_i = paso(tiempo, f_walk[0], g_walk[0], h_walk[0], n_walk[0], t0_walk[0]) # y inicial
    like_walk = append(like_walk,likelihood(campo,y_i)) # likelyhood
   
    iteraciones = 20000
    for i in range(iteraciones):
        f_prime = np.random.normal(f_walk[i],100)# 2.0*dd) 
        g_prime = np.random.normal(g_walk[i],0.5)# 2.0*dd)
        h_prime = np.random.normal(h_walk[i],30.0)# 50)
        n_prime = np.random.normal(n_walk[i],1.0/50.0)# 50)
        t0_prime = np.random.normal(t0_walk[i],5)# 270.0/2.0)
    
        y_init = paso(tiempo, f_walk[i], g_walk[i], h_walk[i], n_walk[i], t0_walk[i])
        y_prime = paso(tiempo, f_prime, g_prime, h_prime, n_prime, t0_prime)
    
        like_prime = likelihood(campo, y_prime)
        like_init = likelihood(campo, y_init)
    
    
        alpha = like_prime-like_init 
    
        if(alpha>=0.0): # likelihood definido sin exp
                f_walk = append(f_walk, f_prime)
                g_walk = append(g_walk, g_prime)
                h_walk = append(h_walk, h_prime)
                n_walk = append(n_walk, n_prime)
                t0_walk = append(t0_walk, t0_prime)
                like_walk = append(like_walk, like_prime)
        else:
            beta = np.random.random()
            if(alpha>=beta):
                f_walk = append(f_walk, f_prime)
                g_walk = append(g_walk, g_prime)
                h_walk = append(h_walk, h_prime)
                n_walk = append(n_walk, n_prime)
                t0_walk = append(t0_walk, t0_prime)
                like_walk = append(like_walk, like_prime)
            else:
                f_walk = append(f_walk, f_walk[i])
                g_walk = append(g_walk, g_walk[i])
                h_walk = append(h_walk, h_walk[i])
                n_walk = append(n_walk, n_walk[i])
                t0_walk = append(t0_walk, t0_walk[i])
                like_walk = append(like_walk, like_walk[i])
                
    return f_walk, g_walk, h_walk, n_walk, t0_walk, like_walk


# In[4]:

#funcion del fit lineal

def fit_lineal(like_walk, f_walk, g_walk, h_walk, n_walk, t0_walk):
    max_likelihood_id = argmax(like_walk)
    best_like = like_walk[max_likelihood_id]
    best_f = f_walk[max_likelihood_id]
    best_g = g_walk[max_likelihood_id]
    best_h = h_walk[max_likelihood_id]
    best_n = n_walk[max_likelihood_id]
    best_t0 = t0_walk[max_likelihood_id]
    best_campo = paso(tiempo, best_f, best_g, best_h, best_n, best_t0)
    return best_f, best_g, best_h, best_n, best_t0, best_campo, best_like


# In[6]:

titulo = 'Ajuste Lineal Paso'
info = "x,y son las coordenas del pixel analizado en el archivo 'hmi.m_45s.magnetogram.subregion_x1y1.fits'. best_b y best_a son los parametros de maximo likelihood para el modelo lineal"
heading = 'x,y,best_likelihood,best_f,best_g,best_h,best_n,best_t0' #modificar cada parametro
f = open('Modelo_Paso.csv','w')
f.write(titulo+'\n')
f.write(info+'\n')
f.write(heading+'\n')


n=1
for i in range(n):
    campo=cols[:,x[i],y[i]]
    
    f_walk, g_walk, h_walk, n_walk, t0_walk, like_walk=mcmc_lineal_paso(tiempo, campo)

    best_f, best_g, best_h, best_n, best_t0, best_campo, best_like=fit_lineal(like_walk, f_walk, g_walk, h_walk, n_walk, t0_walk)

    f.write(str(x[i]) + ',' + str(y[i]) + ',' + '%.7f'%best_like + ',' + '%.7f'%best_f + ',' + '%.7f'%best_g + ',' + '%.7f'%best_h + ',' + '%.7f'%best_n + ',' + '%.7f'%best_t0 +  '\n')
    
    
    #plot(tiempo,best_campo,'g-')
    #plot(tiempo,campo,'bo')
    #show()
f.close()


# In[14]:

get_ipython().system(u'cat Modelo_Paso.csv')


# In[ ]:



