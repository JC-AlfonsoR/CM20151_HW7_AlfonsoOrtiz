
# coding: utf-8

# In[1]:

from pylab import*
import pyfits
from scipy.optimize import curve_fit

datos = pyfits.open('hmi.m_45s.magnetogram.subregion_x1y1.fits')
cols = datos[0].data
#arreglo para el tiempo
tiempo=np.genfromtxt('times_delta.csv', delimiter=',')


# ###Modelo lineal
# $$B(t)=a+Bt$$
# $$a\sim[-200,200]$$
# $$b\sim[-1,1]$$
# Implemento correcciones para mejorar los ajustes:
# $$a_0 =\bar{c}\pm \sigma/2$$
# $$b_0 =\frac{c_n-c_0}{t_n-t_0}\pm \sigma/2$$

# In[2]:

#def modelo lineal
def modelo_lineal(t_obs,a,b):
    return a + b*t_obs

#funcion de likelihood lineal, sin exp
def likelihood(y_obs,y_m):
    chi_cuadrado= (1.0/2.0)*sum((y_obs-y_m)**2) # Se esta asumiendo que los sigmas = 1
    return -chi_cuadrado


# In[4]:

#coordenadas del pixel seleccionado, se seleccionaroon 100

x = [2*i+1 for i in range(100)]
y = [ i +2 for i in range(100)]


# In[11]:

#MCMC modelo lineal

#modelo lineal con función
def mcmc_modelo_lineal(z, campo_m):
    dd = std(campo)
    mm = mean(campo)
    nn = (campo[-1]-campo[0])/(tiempo[-1]-tiempo[0])

    a_walk = empty((0))
    b_walk = empty((0))
    like_walk = empty((0))
    a_walk = np.append(a_walk,np.random.random()*2.0*dd +mm-1.0*dd)
    b_walk = np.append(b_walk,np.random.random()*2.0*dd +nn-1.0*dd)
    
    #y inicial
    y_i = modelo_lineal(z, a_walk[0],b_walk[0])

    #areglo con la función de likelikelihood
    like_walk= np.append(like_walk,likelihood(campo_m,y_i))
    
    iteraciones = 20000
    
    for i in range(iteraciones):
        a_prime = np.random.normal(a_walk[i], 2.0*dd) 
        b_prime = np.random.normal(b_walk[i], 2.0*dd)

        y_init = modelo_lineal(z, a_walk[i], b_walk[i])
        y_prime = modelo_lineal(z, a_prime, b_prime)
    
        like_prime = likelihood(campo_m, y_prime)
        like_init = likelihood(campo_m, y_init)
    
        alpha = like_prime-like_init #ya estan en logaritmicos
        if(alpha>=0.0):
            a_walk  = append(a_walk,a_prime)
            b_walk  = append(b_walk,b_prime)
            like_walk = append(like_walk, like_prime)
        else:
            beta = log(np.random.random())
            if(beta<=alpha):
                a_walk = append(a_walk,a_prime)
                b_walk = append(b_walk,b_prime)
                like_walk = append(like_walk, like_prime)
            else:
                a_walk = append(a_walk,a_walk[i])
                b_walk = append(b_walk,b_walk[i])
                like_walk = append(like_walk, like_init)
                
                
    return a_walk, b_walk, like_walk
    


# In[12]:

#funcion del fit lineal

def fit_lineal(like, a, b):
    max_likelihood_id = argmax(like)
    best_like = like[max_likelihood_id]
    best_a = a[max_likelihood_id]
    best_b = b[max_likelihood_id]
    best_campo = modelo_lineal(tiempo, best_a, best_b)
    
    return best_a, best_b, best_campo, best_like


# In[13]:

#iteracion de los n pixeles

titulo = 'Ajuste Lineal'
info = "x,y son las coordenas del pixel analizado en el archivo 'hmi.m_45s.magnetogram.subregion_x1y1.fits'. best_b y best_a son los parametros de maximo likelihood para el modelo lineal"
heading = 'x,y,best_likelihood,best_a,best_b' #modificar cada parametro
f = open('Modelo_Lineal.csv','w')
f.write(titulo+'\n')
f.write(info+'\n')
f.write(heading+'\n')

n=2
for i in range(n):
    campo=cols[:,x[i],y[i]]
    
    a_walk, b_walk, like_walk=mcmc_modelo_lineal(tiempo, campo)

    best_a, best_b, best_campo, best_like=fit_lineal(like_walk, a_walk, b_walk)
    
    f.write(str(x[i]) + ',' + str(y[i]) + ',' + '%.7f'%best_like + ',' + '%.7f'%best_a + ',' + '%.7f'%best_b + '\n')
    
    #plot(tiempo,best_campo,'g-')
    #plot(tiempo,campo,'bo')
    #show()
f.close()


# In[14]:

get_ipython().system(u'cat Modelo_Lineal.csv')
get_ipython().system(u'wc Modelo_Lineal.csv')


# In[ ]:



