#intento de leer fechas, pero se intentó meor con mupy en python
library(lubridate)
library(dplyr)
library(tidyr)
destfile <- "Metodos2015/HW7/CM20151_HW7_AlfonsoOrtiz/P2_Campo_magnetico_solar/times.csv"
MyData <- read.csv(destfile, header=TRUE, sep=",", skiprow=0)

data=data.frame(MyData)
colnames(data) <- c("fecha","hora")
dat2 <- data.frame(do.call(rbind, strsplit(as.vector(data$hora), split = "_")))
one=dat2[1,2]
dat1=data.frame(dat2[2])

