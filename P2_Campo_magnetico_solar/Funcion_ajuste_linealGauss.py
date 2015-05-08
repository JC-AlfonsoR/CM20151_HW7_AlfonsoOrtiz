
# coding: utf-8

# In[2]:

from pylab import*
import pyfits
from scipy.optimize import curve_fit

datos = pyfits.open('hmi.m_45s.magnetogram.subregion_x1y1.fits')
cols = datos[0].data
#arreglo para el tiempo
tiempo=np.genfromtxt('times_delta.csv', delimiter=',')



# ###Lineal Gaussiano
# $$B(t) = c + dt+\frac{1}{\sigma\sqrt{2\pi}}exp\left(-\frac{1}{2}\left(\frac{t-\mu}{\sigma}\right)^2\right)$$

# In[3]:

#coordenadas del pixel seleccionado

x = [2*i+1 for i in range(100)]
y = [ i +2 for i in range(100)]


#funcion de likelihood lineal, sin exp
def likelihood(y_obs,y_m):
    chi_cuadrado= (1.0/2.0)*sum((y_obs-y_m)**2) # Se esta asumiendo que los sigmas = 1
    return -chi_cuadrado
#funcion lineal_gauss
def lineal_gauss(t_obs, c, d, sigma, mu):
    n1 = sigma**2/(sqrt(2*pi))
    n2 = 0.5*((t_obs-mu)/sigma)**2
    return c +d*t_obs + n1*exp(-n2)


# In[10]:

#MCMC modelo lineal

#modelo lineal con función
def mcmc_modelo_linealGauss(tiempo,campo):
    c_walk = empty((0))
    d_walk = empty((0))
    sigma_walk = empty((0))
    mu_walk = empty((0))
    like_walk = empty((0))

    mm = mean(campo)
    dd = std(campo)
    nn = (campo[-1]-campo[0])/(tiempo[-1]-tiempo[0])
    mm_t = mean(tiempo)

    # Numeros iniciales
    c_walk = append(c_walk,np.random.random()*2.0*dd +mm-1.0*dd)
    d_walk = append(d_walk,np.random.random()*2.0*dd +nn-1.0*dd)
    sigma_walk = append(sigma_walk, np.random.random()*10+35)#*2.0**dd - 1.0*dd)
    mu_walk = append(mu_walk, np.random.random()*10.0 + 95.0)

    #y_inicial
    y_i = lineal_gauss(tiempo, c_walk[0],d_walk[0],sigma_walk[0], mu_walk[0]) # y inicial
    like_walk = append(like_walk,likelihood(campo,y_i)) # likelyhood
    
    iteraciones = 20000
    
    for i in range(iteraciones):
        c_prime = np.random.normal(c_walk[i], 2.0*dd) # Desviacion estandar se asume que es 1 desde la definicion del likelyhood
        d_prime = np.random.normal(d_walk[i], 2.0*dd)
        sigma_prime = np.random.normal(sigma_walk[i], 5)
        mu_prime = np.random.normal(mu_walk[i],5)#mm_t/5.0)

        y_init = lineal_gauss(tiempo, c_walk[i], d_walk[i], sigma_walk[i], mu_walk[i])
        y_prime = lineal_gauss(tiempo, c_prime, d_prime, sigma_prime, mu_prime)

        like_prime = likelihood(campo, y_prime)
        like_init = likelihood(campo, y_init)


        alpha = like_prime-like_init 

        if(alpha>=0.0): # likelihood definido sin exp
                c_walk = append(c_walk,c_prime)
                d_walk = append(d_walk,d_prime)
                sigma_walk = append(sigma_walk,sigma_prime)
                mu_walk = append(mu_walk,mu_prime)
                like_walk = append(like_walk, like_prime)
        else:
            beta = np.random.random()
            if(alpha>=beta):
                c_walk = append(c_walk,c_prime)
                d_walk = append(d_walk,d_prime)
                sigma_walk = append(sigma_walk,sigma_prime)
                mu_walk = append(mu_walk,mu_prime)
                like_walk = append(like_walk, like_prime)
            else:
                c_walk = append(c_walk, c_walk[i])
                d_walk = append(d_walk, d_walk[i])
                sigma_walk = append(sigma_walk,sigma_walk[i])
                mu_walk = append(mu_walk,mu_walk[i])
                like_walk = append(like_walk, like_walk[i])
    return c_walk, d_walk, sigma_walk, mu_walk, like_walk


# In[11]:

def fit_linealGauss(c_walk, d_walk, sigma_walk, mu_walk, like_walk):
    max_likelihood_id = argmax(like_walk)
    best_like = like_walk[max_likelihood_id]
    best_c = c_walk[max_likelihood_id]
    best_d = d_walk[max_likelihood_id]
    best_sigma = sigma_walk[max_likelihood_id]
    best_mu = mu_walk[max_likelihood_id]
    best_campo = lineal_gauss(tiempo, best_c, best_d, best_sigma, best_mu)
    
    return best_c, best_d, best_sigma, best_mu, best_campo, best_like
    


# In[12]:

#iteracion de los n pixeles
titulo = 'Ajuste Lineal Gauss'
info = "x,y son las coordenas del pixel analizado en el archivo 'hmi.m_45s.magnetogram.subregion_x1y1.fits'. best_b y best_a son los parametros de maximo likelihood para el modelo lineal"
heading = 'x,y,best_likelihood,best_c,best_d,best_sigma,best_mu' #modificar cada parametro
f = open('Modelo_Gauss.csv','w')
f.write(titulo+'\n')
f.write(info+'\n')
f.write(heading+'\n')

n=2
for i in range(n):
    campo=cols[:,x[i],y[i]]
    
    c_walk, d_walk, sigma_walk, mu_walk, like_walk = mcmc_modelo_linealGauss(tiempo, campo)

    best_c, best_d, best_sigma, best_mu, best_campo, best_like = fit_linealGauss(c_walk, d_walk, sigma_walk, mu_walk, like_walk) 
    
    f.write(str(x[i]) + ',' + str(y[i]) + ',' + '%.7f'%best_like + ',' + '%.7f'%best_c + ',' + '%.7f'%best_d + ',' + '%.7f'%best_sigma + ',' + '%.7f'%best_mu +  '\n')
    
    plot(tiempo,best_campo,'g-')
    plot(tiempo,campo,'bo')
    show()
f.close()


# In[15]:

get_ipython().system(u'cat Modelo_Gauss.csv')


# In[ ]:



