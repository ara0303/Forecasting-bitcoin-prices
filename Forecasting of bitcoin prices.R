library("anytime")
library("bsts")
library("car")
library("caret")
library("forecast")
library("keras")
library("MCMCpack")
library("smooth")
library("tensorflow")
library("tseries")
library("TTR")
library("rnn")

#Importing data
setwd("E:/Fall 2017/data analytics/project")
train <- read.csv("train.csv", header = T, stringsAsFactors = F)
test <- read.csv("test.csv", header = T, stringsAsFactors = F)
summary(train)
testdata <- test[,2] #test data
head(Train)
#Converting data for analysis
train$Date <- as.Date(anytime(train$Date))
test$Date <- as.Date(anytime(test$Date))
train$Volume <- gsub(",", "", train$Volume)
train$Market.Cap <- gsub(",", "", train$Market.Cap)
train$Market.Cap <- as.numeric(train$Market.Cap)
train$Volume <- as.numeric(train$Volume)

#Difference between high and low on each day
a <- matrix(c(0), nrow = 0, ncol = 1)
for(i in 1:nrow(train)){
  a <- rbind(a, train[i,3] - train[i,4])
  i <- i + 1
}
train <- cbind(train,a)


#Volume has missing values#
#Data Manipulation#
fifty_avg <- round(mean(train$Volume[train$a < 50], na.rm = TRUE), digits = 2)
hun_avg <- round(mean(train$Volume[train$a > 50 & train$a < 100], na.rm = TRUE), digits = 2)
hf_avg <- round(mean(train$Volume[train$a > 100 & train$a < 150], na.rm = TRUE), digits = 2)
th_avg <- round(mean(train$Volume[train$a > 150 & train$a < 350], na.rm = TRUE), digits = 2)
for(i in 1:nrow(train)){
  if(is.na(train[i,6])){
    if(train$a[i] < 50){
      train$Volume[i] <- fifty_avg
    } else if(train$a[i] < 100){
      train$Volume[i] <- hun_avg
    } else if(train$a[i] < 150){
      train$Volume[i] <- hf_avg
    } else if(train$a[i] < 350){
      train$Volume[i] <- th_avg
    }else
      print("Uncaught Title")
  }
}
train <- train[, - 8] #Removing column 8
ggplot(train, aes(Date, Close)) + geom_line() + scale_x_date("year") + ylim(0,10000) + ylab("Closing Price") 


#Convert data set to time series
Train <- xts(train[, -1], order.by = as.POSIXct(train$Date)) 
head(Train)
plot(Train$Close,type='l',lwd = 1.5,col='red', ylim = c(0,10000))

#Time series of closing price
tsr <- ts(Train[,4], frequency = 365.25,start = c(2013,4,27))
plot.ts(tsr, ylim = c(0,10000), main = "Bitcoin Closing Price")


#checking for trends and seasonality
dects <- decompose(tsr) #Obtaining the trends and seasonality
plot(dects)
des <- seasadj(dects)
plot(des)


#Holt forecasting
holtt <-  holt(Train[1289:1655,'Close'], type = "additive", damped = F) #holt forecast values
plot(holtt, type = "o", xlab = "Time", ylab = "Closing price of bitcoin")
holtf <- forecast(holtt, h = 10)
holtdf <- as.data.frame(holtf)
accuracy(holtdf[,1], testdata)
plot(holtf, ylim = c(0,10000)) 
holtfdf <- cbind(test, holtdf[,1])
ggplot() + geom_line(data = holtfdf, aes(Date, holtfdf[,2]), color = "blue") + geom_line(data = holtfdf, aes(Date, holtfdf[,3]), color = "Dark Red")

#Triple exponential smoothing forecasting
ETS <- ets((Train[,'Close'])) # ETS forecast values
plot(ETS, type = "o", xlab = "Time", ylab = "Closing price of bitcoin")
ETSf <- forecast(ETS, h = 10)
etsdf <- as.data.frame(ETSf)
accuracy(etsdf[,1], testdata)
plot(forecast(ETS, h = 10), ylim = c(0,10000)) #ETS forecast plot works perfectly
etsp <- predict(ETS, n.ahead = 10, prediction.interval = T, level = 0.95)

#Holt-Winter Forecasting
a <- HoltWinters(Train[,4], gamma = F)
plot(a, type = "o", xlab = "Time", ylab = "Closing price of btc")
HWf <- forecast(a, h = 10)
accuracy(HWf)
HWdf <- as.data.frame(HWf) #Works well
accuracy(HWdf[,1], testdata)
plot(forecast(a,h=10), ylim = c(0,10000))
asa <- predict(a, n.ahead = 10, prediction.interval = T, level = 0.95)
plot(a,asa)


#ARIMA forecasting 
tsdf <- diff(Train[,4], lag = 2)
tsdf <- tsdf[!is.na(tsdf)]
adf.test(tsdf)
plot(tsdf, type = 1, ylim = c(-1000, 1000))
acf(tsdf)
pacf(tsdf)
gege <- arima(Train[,4], order = c(4,2,11))
gegef <- as.data.frame(forecast(gege, h = 10))
accuracy(gegef[,1], testdata)
gegefct <- cbind(test, gegef[,1])
ggplot() + geom_line(data = gegefct, aes(Date, gegefct[,2]), color = "blue") + geom_line(data = gegefct, aes(Date, gegefct[,3]), color = "Dark Red")

#Cross validating with best autofit arima
arimaa <- auto.arima(Train[,4]) # Arima model values
arimaf <- as.data.frame(forecast(arimaa, h = 10))
plot(forecast(arimaa, h = 10), ylim = c(0,10000)) #arima forecast plot for next 10 days
plot(arimaf[,1], testdata)
accuracy(arimaf[,1], testdata)

#forecasting with mlr
aropen <- auto.arima(Train[,1])
aropenf <- as.data.frame(forecast(aropen, h = 10))
arhigh <- auto.arima(Train[,2])
arhighf <- as.data.frame(forecast(arhigh, h = 10))
arlow <- auto.arima(Train[,3])
arlowf <- as.data.frame(forecast(arlow, h = 10))
arvol <- auto.arima(Train[,5])
arvolf <- as.data.frame(forecast(arvol, h = 10))
armc <- auto.arima(Train[,6])
armcf <- as.data.frame(forecast(armc, h = 10))
reg <- lm(Close~., data = Train)
summary(reg)
closepr <- matrix(c(0), nrow = 0, ncol = 1)
closelow <- matrix(c(0), nrow = 0, ncol = 1)
closehigh <- matrix(c(0), nrow = 0, ncol = 1)
#Use this regression equation to predict the closing price based on forecasted values of the rest of the variables
for(i in 1:10){
Closep <- -0.5315 + (-0.5077 * aropenf[i,1]) + (0.7765 * arhighf[i,1]) + (0.7453 * arlowf[i,1]) + (9.597e-09 * arvolf[i,1]) + (-9.217e-10 * armcf[i,1])  
Closelo <- -0.5315 + (-0.5077 * aropenf[i,2]) + (0.7765 * arhighf[i,2]) + (0.7453 * arlowf[i,2]) + (9.597e-09 * arvolf[i,2]) + (-9.217e-10 * armcf[i,2])  
Closehi <- -0.5315 + (-0.5077 * aropenf[i,3]) + (0.7765 * arhighf[i,3]) + (0.7453 * arlowf[i,3]) + (9.597e-09 * arvolf[i,3]) + (-9.217e-10 * armcf[i,3])  
closepr <- rbind(closepr, Closep)
closelow <- rbind(closelow, Closelo)
closehigh <- rbind(closehigh, Closehi)
i = i + 1
}
closepr #Predicted closing price based on forecasted values of other variables
closelow #95% CI for the bitcoin price
closehigh
accuracy(closepr[,1], testdata)
mlrf <- cbind(test, closepr[,1],closelow[,1], closehigh[,1])
ggplot() + geom_line(data = mlrf, aes(Date, mlrf[,2]), color = "blue") + geom_line(data = mlrf, aes(Date, mlrf[,3]), color = "Dark Red") + geom_line(data = mlrf, aes(Date, mlrf[,4]), color = "Green") + geom_line(data = mlrf, aes(Date, mlrf[,5]), color = "Black")
 

#Bayesian forecasting
library(bsts)
ss <- AddLocalLinearTrend(list(), Train[,4]) #Adding linear trend to model
ss <- AddSeasonal(ss, Train[,4], nseasons = 365.25) #Adding seasonal trend to model
model1 <- bsts(Train[,4],
               state.specification = ss,
               niter = 10)

plot(model1, ylim = c(0,10000)) #Plot based on bayesian regression of the model
pred1 <- predict(model1, horizon = 10)
plot(pred1, plot.original = 50,ylim = c(0,9000))
pred1$mean
accuracy(pred1$mean, testdata)

model2 <- bsts(Close ~ ., state.specification = ss,
               niter = 10,
               data = as.data.frame(Train))
model3 <- bsts(Close ~ ., state.specification = ss,
               niter = 10,
               data = as.data.frame(Train),
               expected.model.size = 10)
CompareBstsModels(list("Model 1" = model1, "Model 2" = model2, "Model 3" = model3), colors = c("blue", "red", "green"))
                               


