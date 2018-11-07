library(ggplot2)
library(corrplot)
library(pROC)
library(xgboost)
library(Matrix)
library(caret)
library(dummies)
library(dummy)
library(randomForest)
library(e1071)
library(dplyr)

#Transform data
dataset <- read.csv('sources/default_of_credit_card_clients.csv', sep = ";", header = TRUE) %>%
  .[, -c(1)]
dataset["class"] <- dataset[, "default.payment.next.month"]
dataset["class"] <- factor(dataset$class,
       levels = sort(unique(dataset$class)),
       labels = c("Good", "Bad"))
dataset$default.payment.next.month <- NULL

dataset["MARRIAGE"] <- factor(dataset$MARRIAGE,
                           levels = sort(unique(dataset$MARRIAGE)),
                           labels = c("UNKNOWN", "MARRIED", "SINGLE", "OTHERS"))

dataset["EDUCATION"] <- factor(dataset$EDUCATION,
                           levels = sort(unique(dataset$EDUCATION)),
                           labels = c("UNKNOWN", "GRADUATE_SCHOOL", "UNIVERSITY", "HIGH_SCHOOL", "OTHERS", "UNKNOWN", "UNKNOWN"))

dataset["SEX"] <- factor(dataset$SEX,
                              levels = sort(unique(dataset$SEX)),
                              labels = c("MALE", "FEMALE"))

#check missing values
sum(sapply(dataset, function(y) sum(is.na(y) | is.null(y)))) != 0


################# Skip dealing with outliers #########################

######################################################################


###### Correlation

# Continuous variables
ggplot(data = melt(dataset[, -c(2:4)],id.vars = "class", variable.name = "field")) + 
  geom_boxplot(aes(x = class, y=value, color = class)) + 
  facet_wrap(~field,scale = "free") +
  scale_color_manual(values= c("green", "red")) +
  theme(legend.position = "none")

keep_field <- c("LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4")

box_tops <- dataset[1, c("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", 
                         "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6")]
data_to_plot <- dataset[ , c("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", 
                             "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "class")]
lapply(colnames(box_tops), function(field){
  if(field != "class"){
    IQR = quantile(dataset[, field])[4] - quantile(dataset[, field])[2]
    box_tops[, field] <<- quantile(dataset[, field])[3] + 1.5 * IQR
    
    data_to_plot <<- data_to_plot %>%
      filter(get(field) <= box_tops[1, field])
  }  
})

ggplot(data = melt(data_to_plot,id.vars = "class", variable.name = "field")) + 
  geom_boxplot(aes(x = class, y=value, color = class)) + 
  facet_wrap(~field,scale = "free") +
  scale_color_manual(values= c("green", "red")) +
  theme(legend.position = "none")

keep_field <- append(keep_field, c("PAY_AMT1", "PAY_AMT2", "PAY_AMT3")) # keep discriminant features 

# Categorical variables
chisq.test(dataset$SEX,dataset$class)
chisq.test(dataset$EDUCATION,dataset$class)
chisq.test(dataset$MARRIAGE,dataset$class)

prop.table(table(dataset$SEX,dataset$class), 1) 
prop.table(table(dataset$EDUCATION,dataset$class), 1) 
prop.table(table(dataset$MARRIAGE,dataset$class), 1)

#correlation with SEX, EDUCATION AND MARRIAGE
keep_field <- append(keep_field, c("SEX", "EDUCATION", "MARRIAGE"))

# Build the correlated dataset
corr_dataset <- dataset[, append(keep_field, "class")]

#####################

#build the train and test sets
set.seed(1)


#used afterwards to create a 50/50 class train set
# get 5000 of the 6636 rows of the "1 class" for the train and add 5010 "0 class" rows
saved_dataset <- corr_dataset
temp_dataset <- corr_dataset[corr_dataset$class == 1, ][1:5000,] 

eval <- sample(1:nrow(corr_dataset[corr_dataset$class == 0, ]), nrow(temp_dataset) + 10)
temp_dataset <- rbind(temp_dataset, corr_dataset[corr_dataset$class == 0, ][eval,])
temp_true_dataset <- corr_dataset[corr_dataset$class == 0, ][-eval,]
temp_true_dataset <- rbind(temp_true_dataset, corr_dataset[corr_dataset$class == 1, ][5001: nrow(corr_dataset[corr_dataset$class == 1, ]),]) 

eval <- sample(1:nrow(temp_dataset), 0.35*nrow(temp_dataset))
test <- temp_dataset[eval,]
train <- temp_dataset[-eval,]

# used for the first model, respect the ratio of "1" and "0" class for the train/test set
corr_dataset$class = factor(corr_dataset$class, labels = c(0, 1))
eval <- sample(1:nrow(corr_dataset), 0.35*nrow(corr_dataset))
test <- corr_dataset[eval,]
train <- corr_dataset[-eval,]


ggplot(corr_dataset) +
  geom_bar(aes(x = class)) # Unbalanced data => 75% of "0" and 25% of "1"


# cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1, number = 3)
# 
# rf.grid <- expand.grid(nrounds = 500,
#                         eta = c(0.01,0.05,0.1),
#                         max_depth = c(2,4,6),
#                         min_child_weight = c(1, 2, 5),
#                         colsample_bytree = 1,
#                         gamma = 1,
#                         subsample = .8)
# 
# xgb_tune <- train(class ~ .,
#                   data=train,
#                   method="xgbTree",
#                   trControl=cv.ctrl,
#                   tuneGrid=xgb.grid,
#                   verbose=T,
#                   metric="Kappa",
#                   nthread =3)


# - RandomForest
RF_classifier <- randomForest(class ~ ., data = temp_dataset)

RF_predicted <- predict(bestmtry, temp_true_dataset, type = "class")
confusionMatrix(data = RF_predicted, reference = temp_true_dataset$class, positive = "0")

rf_roc_curve <- roc(test$class, as.integer(as.vector(RF_predicted)))
plot(rf_roc_curve)

# - Logistic regression
#dummy des features pour être utilisés par le glm
train$class <- as.integer(as.vector(train$class))
train <- dummy.data.frame(train)
train$class <- as.factor(train$class)

test$class <- as.integer(as.vector(test$class))
test <- dummy.data.frame(test)
test$class <- as.factor(test$class)

lr_classifier <- glm(class ~ ., family = binomial(), data = temp_dataset)
lr_predicted <- predict(lr_classifier, temp_true_dataset, type = "response")
confusionMatrix(data = factor(ifelse(lr_predicted>0.5, 1,0), levels = c(0, 1)), reference = temp_true_dataset$class, positive = "0")

lr_roc_curve <- roc(temp_true_dataset$class, as.integer(as.vector(factor(ifelse(lr_predicted>0.5, 1,0), levels = c(0, 1)))))
plot(lr_roc_curve)


# - Gradient boosting
xgb_params_multi <- list(objective = "multi:softmax",
                   num_class = 2,
                   max_depth = 2,
                   eta = 0.05, 
                   gamma = 1, 
                   colsample_bytree = 1, 
                   min_child_weight = 5, 
                   subsample = 0.8)

xgb_params_logi <- list(objective = "binary:logistic",
                   max_depth = 2,
                   eta = 0.05, 
                   gamma = 1, 
                   colsample_bytree = 1, 
                   min_child_weight = 5, 
                   subsample = 0.8)

# corr_dataset$class = factor(corr_dataset$class, labels = c("Good", "Bad"))
# eval <- sample(1:nrow(corr_dataset), 0.35*nrow(corr_dataset))
# test <- corr_dataset[eval,]
# train <- corr_dataset[-eval,]

# cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, classProbs = TRUE, allowParallel=T)
# 
# xgb.grid <- expand.grid(nrounds = 500,
#                         eta = c(0.01,0.05,0.1),
#                         max_depth = c(2,4,6),
#                         min_child_weight = c(1, 2, 5),
#                         colsample_bytree = 1,
#                         gamma = 1,
#                         subsample = .8)
# 
# xgb_tune <- train(class ~ .,
#                   data=train,
#                   method="xgbTree",
#                   trControl=cv.ctrl,
#                   tuneGrid=xgb.grid,
#                   verbose=T,
#                   metric="Kappa",
#                   nthread =3)

#predicted <- predict(xgb_tune, test[, -c(20)], type = "prob")
# confusionMatrix(data = factor(ifelse(predicted$Bad>0.5, "Bad","Good"), levels = c("Good", "Bad")), reference = test$class, positive = "Good")


bestmtry <- tuneRF(temp_dataset[, -c(12)], temp_dataset[, 12], stepFactor=1.5, improve=1e-5, ntree=500)

temp_dataset$class <- as.integer(as.vector(temp_dataset$class))
temp_dataset <- dummy.data.frame(temp_dataset)
temp_dataset$class <- as.factor(temp_dataset$class)

temp_true_dataset$class <- as.integer(as.vector(temp_true_dataset$class))
temp_true_dataset <- dummy.data.frame(temp_true_dataset)
temp_true_dataset$class <- as.factor(temp_true_dataset$class)

gb_classifier <- xgboost(data = xgb.DMatrix(as.matrix(temp_dataset[, -c(20)]), label=as.integer(as.vector(temp_dataset$class))), params = xgb_params_multi, nrounds = 1000)

gb_predicted <- predict(gb_classifier, xgb.DMatrix(as.matrix(temp_true_dataset[, -c(20)]), label=as.integer(as.vector(temp_true_dataset$class))), type = "response")

confusionMatrix(data = as.factor(gb_predicted), reference = temp_true_dataset$class, positive = "0")

gb_roc_curve <- roc(temp_true_dataset$class, as.integer(as.vector(gb_predicted)))
plot(gb_roc_curve)

#SVM

tuneResult1 <- tune(svm, class ~ .,  data = train, kernel = "radial",  scale = T, proba = F,
                    ranges = list(epsilon = c(0.05, 0.1), cost = c(0.01, 0.1, 1, 5)))


svm_classifier <- svm(class ~ ., data = train, kernel = "radial", cross = 10, cost = 0.1, gamma = .5)
predicted <- predict(tuneResult1$best.model, test, type = "class")

confusionMatrix(data = predicted, reference = test$class, positive = "0")

svm_roc_curve <- roc(test$class, as.integer(as.vector(predicted)))

#tableau comparatif

# comparating <- data.frame(gb_roc_curve$auc, gb_roc_curve$sensitivities[2], gb_roc_curve$specificities[2])
# colnames(comparating) = c("AUC", "sensitivity", "specitivity")
# rownames(comparating)[1] = "xgboost"
# plot(gb_roc_curve)
# 
# compar_lr <- data.frame(lr_roc_curve$auc, lr_roc_curve$sensitivities[2], lr_roc_curve$specificities[2])
# colnames(compar_lr) = c("AUC", "sensitivity", "specitivity")
# rownames(compar_lr)[1] = "logistic regression"
# comparating <- rbind(comparating, compar_lr)
# 
# compar_rf <- data.frame(rf_roc_curve$auc, rf_roc_curve$sensitivities[2], rf_roc_curve$specificities[2])
# colnames(compar_rf) = c("AUC", "sensitivity", "specitivity")
# rownames(compar_rf)[1] = "Random Forest"
# comparating <- rbind(comparating, compar_rf)
# 
# comparating

ggplot(data = data, aes(x = LIMIT_BAL)) +
  geom_density()

preprocessor <- preProcess(corr_dataset, method = c("center", "scale", "BoxCox"))
data.prep <- predict(preprocessor, corr_dataset)
data.prep$class <-corr_dataset[, "class"]

ggplot(data = data.prep, aes(x = LIMIT_BAL)) +
  geom_density()

# ggplot(data = melt(data.prep[, -c(9:11)],id.vars = "class", variable.name = "field")) + 
#   geom_boxplot(aes(x = class, y=value, color = class)) + 
#   facet_wrap(~field,scale = "free") +
#   scale_color_manual(values= c("green", "red")) +
#   theme(legend.position = "none")

# data.prep <- data.prep[, c("PC1", "PC2", "class")]
test <- data.prep[eval,]
train <- data.prep[-eval,]

# ggplot(data = data.prep, aes(x = PC1, y = PC2, color = class)) +
#   geom_point()

svm_classifier <- svm(class ~ ., data = temp_dataset, kernel = "radial", epsilon = .05, cost = 1, gamma = .052)
predicted <- predict(svm_classifier, temp_true_dataset, type = "class")

confusionMatrix(data = predicted, reference = temp_true_dataset$class, positive = "0")

svm_roc_curve <- roc(test$class, as.integer(as.vector(predicted)))


### tentative PCA avec XGboost


######## Encode the train dataset
# pca.model <- princomp(train[train$class == 0, -c(9:12)])
# 
# pca.train <- predict(pca.model, train[, -c(9:12)])
# obs.train <- train[, 1:8]
# colnames(pca.train) <- colnames(train[,1:8])
# diff.train <- obs.train - pca.train
# diff.train$error <- rep(0, nrow(diff.train))
# for (i in 1:nrow(diff.train)){
#   diff.train[i, "error"] <- sqrt(sum(diff.train[i,]^2))
# }
# diff.train$class <- train$class
# 
# ggplot(data = diff.train[diff.train$error < 180000, ][1:1000,] , aes(x = rownames(diff.train[diff.train$error < 180000, ])[1:1000], y = error, col = class)) +
#   geom_point()
# 
# xgb_params_multi <- list(objective = "multi:softmax",
#                          num_class = 2,
#                          max_depth = 2,
#                          eta = 0.05, 
#                          gamma = 1, 
#                          colsample_bytree = 1, 
#                          min_child_weight = 5, 
#                          subsample = 0.8)
# 
# xgb_params_logi <- list(objective = "binary:logistic",
#                         max_depth = 2,
#                         eta = 0.05, 
#                         gamma = 1, 
#                         colsample_bytree = 1, 
#                         min_child_weight = 5, 
#                         subsample = 0.8)
# 
# gb_classifier <- xgboost(data = xgb.DMatrix(as.matrix(train[, -c(3)]), label=as.integer(as.vector(train$class))), params = xgb_params_multi, nrounds = 1000)
# 
# gb_predicted <- predict(gb_classifier, xgb.DMatrix(as.matrix(test[, -c(3)]), label=as.integer(as.vector(test$class))), type = "response")
# 
# confusionMatrix(data =  factor(ifelse(gb_predicted>0.5, 1,0), levels = c(0, 1)), reference = test$class, positive = "0")