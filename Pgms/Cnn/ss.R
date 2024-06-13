library(EBImage)
library(keras)
library(tensorflow)
install_keras()
install_tensorflow()
setwd('C:/Users/Aravind Prasad/Desktop/MRICE/Train')
pics<-c('Bacterial1.jpg','Bacterial2.jpg','Bacterial3.jpg','Bacterial4.jpg','Bacterial5.jpg',
        'Bacterial6.jpg','Bacterial7.jpg','Bacterial8.jpg','Bacterial9.jpg','Brown1.jpg','Brown2.jpg',
        'Brown3.jpg','Brown4.jpg','Brown5.jpg','Brown6.jpg','Brown7.jpg','Brown8.jpg','Brown9.jpg')
mypic<-list()
for(i in 1:18){
  mypic[[i]] <- readImage(pics[i])
}
print(mypic[[1]])
display(mypic[[2]])
summary(mypic[[1]])
hist(mypic[[2]])
str(mypic)
for(i in 1:18)
{
  mypic[[i]] <-resize(mypic[[i]],28,28)
}
for(i in 1:18)
{
  mypic[[i]]<-array_reshape(mypic[[i]],c(28,28,3))
}
str(mypic)
trainx<-NULL
for(i in 1:10)
{
  trainx <- rbind(trainx,mypic[[i]])
}
str(trainx)
testx<-rbind(mypic[[6]],mypic[[12]])
trainy <-c(0,0,0,0,0,1,1,1,1,1)
testy<-c(0,1)
trainLabels <- to_categorical(trainy)
testLabels<-to_categorical(testy)
library(keras)
model <- keras_model_sequential()
model %>%
  layer_dense(units =256,activation = 'relu',input_shape=c(2352)) %>%
  layer_dense(units=128,activation ='relu') %>%
  layer_dense(units=2,activation='softmax')
summary(model)
model %>%
  compile(loss='binary_crossentropy',optimizer=optimizer_rmsprop(),metrics=c('accuracy'))
history <- model %>%
  fit(trainx,trainLabels,epochs=30,batch_size = 32,validation_split=0.2)
plot(history)
model %>% evaluate(trainx,trainLabels)
pred <- model %>% predict_classes(trainx)
table(Predicted = pred ,Actual = trainy)
prob <- model %>% predict_proba(trainx)
cbind(prob,Prected = pred,Actual=trainy)
model %>% evaluate(testx,testLabels)
pred <- model %>% predict_classes(testx)
table(Predicted = pred ,Actual =testy)
display(mypic[[6]])
fit(trainx,
    trainLabels,epochs=30,
    batch_size = 32,
    validation_split=0.2)
plot(history)
model %>% evaluate(trainx,trainLabels)
pred <- model %>% predict_classes(trainx)
table(Predicted = pred ,Actual = trainy)
prob <- model %>% predict_proba(trainx)
cbind(prob,Prected = pred,Actual=trainy)
model %>% evaluate(testx,testLabels)
pred <- model %>% predict_classes(testx)
table(Predicted = pred ,Actual =testy)
display(mypic[[6]])
