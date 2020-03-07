labels <- read.table("batches.meta.txt")
images.rgb <- array(dim = c(50000,32,32,3))
images.lab <- matrix(nrow=50000)
num.images = 10000 # Set to 10000 to retrieve all images per file to memory

# Cycle through all 5 binary files
for (f in 1:5) {
  to.read <- file(paste("C:/Users/jj reddy/Desktop/TAMU/SEM 1 - Course work/2019 - Fall/613/Project/Data For Project/data_batch_", f, ".bin", sep=""), "rb")
  for(i in 1:num.images) {
    l <- readBin(to.read, integer(), size=1, n=1, endian="big")
    r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    r1 <- matrix(unlist(r), 32,32, byrow =TRUE)
    g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    g1 <- matrix(unlist(g), 32, 32, byrow =TRUE)
    b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    b1 <- matrix(unlist(b), 32, 32, byrow =TRUE)
    index <- num.images * (f-1) + i
    images.rgb[index,,,] = array(c(r1 ,g1, b1), dim=c(32,32,3))
    images.lab[index,1] = l
  }
  close(to.read)
  remove(l,r,g,b,f,i,index, to.read)
}



install.packages("keras")
install.packages("tensorflow")
install_keras()
install_tensorflow()
library(keras)
library(tensorflow)
devtools::install_github("rstudio/keras")
install_tensorflow(package_url = "https://pypi.python.org/packages/b8/d6/af3d52dd52150ec4a6ceb7788bfeb2f62ecb6aa2d1172211c4db39b349a2/tensorflow-1.3.0rc0-cp27-cp27mu-manylinux1_x86_64.whl#md5=1cf77a2360ae2e38dd3578618eacc03b")
library(tensorflow)
# Parameters --------------------------------------------------------------

batch_size <- 64
epochs <- 100
data_augmentation <- TRUE


-------------------------------------------------------------------------------------------
set.seed(1)
size <- sample(50000,40000)
#tr_sample <- sample(12000, 10000)
partialdata<- images.rgb[,,,]
x_train_1 <- partialdata[size,,,]/255
x_test_1 <- partialdata[-size,,,]/255
partial_labels<- images.lab[,]
y_train_1 <- as.matrix(partial_labels[size])
y_test_1 <- as.matrix(partial_labels[-size])
y_train_1 <- to_categorical(y_train_1, num_classes = 10)
y_test_1 <- to_categorical(y_test_1, num_classes = 10)


# Defining Model ----------------------------------------------------------

# Initialize sequential model
model <- keras_model_sequential()

model %>%
  
  # Start with hidden 2D convolutional layer being fed 32x32 pixel images
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same",input_shape = c(32, 32, 3)) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # 2 additional hidden 2D convolutional layers
  layer_conv_2d(filter = 64, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  
  layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  
  # Use max pooling once more
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.35) %>%
  
  # 3rd additional hidden 2D convolutional layers
  layer_conv_2d(filter = 128, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  
  layer_conv_2d(filter = 128, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  
  # Use max pooling once more
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.45) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  #layer_batch_normalization() %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(10) %>%
  layer_activation("softmax")
  
  opt <- optimizer_rmsprop(lr = 0.001, decay = 1e-6)
  #opt <- optimizer_adam(r = 0.001, decay = 1e-6)
  
  model %>% 
    compile(loss = "categorical_crossentropy",
    optimizer = opt, metrics = "accuracy")


# Saving ----------------------------------------------------------------
  
save_model_hdf5(model, 'first_cnn_model.h5')

##fittting the model-------------------------------------------------------------------
model %>% fit(
  x_train_1, y_train_1,
  batch_size = batch_size,
  steps_per_epoch = as.integer(40000/batch_size),
  epochs = epochs,
  validation_data = list(x_test_1, y_test_1),
  shuffle = TRUE,
  verbose = 2)



##loading the model-----------------------------------------------------------------------
results <- model %>% evaluate(x_test_1, y_test_1, verbose =2)
print(paste0("test loss:",results$loss))
print(paste0("test loss:",results$accuracy))








