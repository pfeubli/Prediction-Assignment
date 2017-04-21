# Prediction Assignment Writeup
PF  
21 4 2017  



## Executive Summary

One thing that people regularly do is quantify how much of a particular activity they do, 
but they rarely quantify how well they do it. In this excercise, we are asked to predict the
way 6 participants performed barbell lifts by using machine learning methods and data from 
accelerometers on the belt, forearm, arm, and dumbell. Barbell lifts were performed in 5
different fashions:

- exactly according to the specification (Class A)
- throwing the elbows to the front (Class B)
- lifting the dumbbell only halfway (Class C) 
- lowering the dumbbell only halfway (Class D)
- throwing the hips to the front (Class E).

In the following analysis, a random forest approach was used to predict the way participants
performed barbell lifts. It predicts with over 99 percent accuracy.

## Data and Data Preprocessing

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral 
Dumbbell Biceps Curl in five different fashions. Body sensors delivered biometric data. The
data set consists of 160 variables and over 19000 observations. The data set is known as
WLE dataset (Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative 
Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference 
in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013).

For analysis, the data was downloaded and #DIV/0! as well as empty cells replaced by NA.

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
TrainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestingUrl  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
TrainingFile<-"pml-traininig.csv"
TestingFile<-"pml-testing.csv"

if(!file.exists(TrainingFile)){
        download.file(TrainingUrl,destfile = TrainingFile)
}
training <- read.csv(TrainingFile, na.strings=c("#DIV/0!", "", "NA"), stringsAsFactors = FALSE)
if(!file.exists(TestingFile)){
        download.file(TestingUrl,destfile = TestingFile)
}
testing  <- read.csv(TestingFile, na.strings=c("#DIV/0!", "", "NA"), stringsAsFactors = FALSE)
```

Then, the training set was split into a training and testing set to apply cross validation
for a better estimate of the test set accuracy.


```r
set.seed(123)
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
TrainingSet <- training[inTrain,]
TestingSet <- training[-inTrain,]
dim(TrainingSet)
```

```
## [1] 13737   160
```

```r
dim(TestingSet)
```

```
## [1] 5885  160
```

The goal of the analysis is to predict the way the exercise was executed (i.e., to predict 
the variable classe). One approach to do this is to use the data from the body sensors as 
in the paper mentioned above. The corresponding variables ar Euler angles (roll, pitch, yaw), 
raw accelerometer, gyroscope and magnetometer.

```r
relvar <- c(grep("^accel", names(training)), grep("^gyros", names(training)), grep("^magnet", 
        names(training)), grep("^roll", names(training)), grep("^pitch", names(training)), 
        grep("^yaw", names(training)), grep("^total", names(training)))
which(colnames(TrainingSet) == "classe")
```

```
## [1] 160
```

```r
TrainingSet <- TrainingSet[, c(relvar, 160)]
TestingSet <- TestingSet[, c(relvar, 160)]
```
A quick check shows that the remaining training set has no variables with low variability
left, which would be bad predictors. 

```r
nearZeroVar(TrainingSet[,-53], saveMetrics = TRUE)
```

```
##                      freqRatio percentUnique zeroVar   nzv
## accel_belt_x          1.068519     1.1720172   FALSE FALSE
## accel_belt_y          1.122109     0.9973065   FALSE FALSE
## accel_belt_z          1.055921     2.1183665   FALSE FALSE
## accel_arm_x           1.133929     5.5761811   FALSE FALSE
## accel_arm_y           1.165605     3.7999563   FALSE FALSE
## accel_arm_z           1.068182     5.6052996   FALSE FALSE
## accel_dumbbell_x      1.031111     2.9555216   FALSE FALSE
## accel_dumbbell_y      1.052023     3.2758244   FALSE FALSE
## accel_dumbbell_z      1.159509     2.8972847   FALSE FALSE
## accel_forearm_x       1.047619     5.6780957   FALSE FALSE
## accel_forearm_y       1.000000     7.1121788   FALSE FALSE
## accel_forearm_z       1.185185     4.0474631   FALSE FALSE
## gyros_belt_x          1.072034     0.9245104   FALSE FALSE
## gyros_belt_y          1.153819     0.4658950   FALSE FALSE
## gyros_belt_z          1.064205     1.2011356   FALSE FALSE
## gyros_arm_x           1.010753     4.6152726   FALSE FALSE
## gyros_arm_y           1.490411     2.6643372   FALSE FALSE
## gyros_arm_z           1.056848     1.7034287   FALSE FALSE
## gyros_dumbbell_x      1.006961     1.7107083   FALSE FALSE
## gyros_dumbbell_y      1.286064     1.9218170   FALSE FALSE
## gyros_dumbbell_z      1.058962     1.4340831   FALSE FALSE
## gyros_forearm_x       1.074792     2.0601296   FALSE FALSE
## gyros_forearm_y       1.029412     5.2485987   FALSE FALSE
## gyros_forearm_z       1.137313     2.1402053   FALSE FALSE
## magnet_belt_x         1.108871     2.2130014   FALSE FALSE
## magnet_belt_y         1.103604     2.0819684   FALSE FALSE
## magnet_belt_z         1.002924     3.1520710   FALSE FALSE
## magnet_arm_x          1.101695     9.6527626   FALSE FALSE
## magnet_arm_y          1.016129     6.2240664   FALSE FALSE
## magnet_arm_z          1.012987     9.1140715   FALSE FALSE
## magnet_dumbbell_x     1.068376     7.7891825   FALSE FALSE
## magnet_dumbbell_y     1.257812     6.0202373   FALSE FALSE
## magnet_dumbbell_z     1.021429     4.7972629   FALSE FALSE
## magnet_forearm_x      1.000000    10.6136711   FALSE FALSE
## magnet_forearm_y      1.163934    13.3726432   FALSE FALSE
## magnet_forearm_z      1.000000    11.8439252   FALSE FALSE
## roll_belt             1.034375     8.0876465   FALSE FALSE
## roll_arm             47.560000    17.3618694   FALSE FALSE
## roll_dumbbell         1.009804    86.3143336   FALSE FALSE
## roll_forearm         11.118367    13.6274296   FALSE FALSE
## pitch_belt            1.204918    12.1715076   FALSE FALSE
## pitch_arm            76.741935    20.2882725   FALSE FALSE
## pitch_dumbbell        2.000000    84.3270001   FALSE FALSE
## pitch_forearm        61.886364    18.9997816   FALSE FALSE
## yaw_belt              1.077348    13.1542549   FALSE FALSE
## yaw_arm              30.487179    19.1089758   FALSE FALSE
## yaw_dumbbell          1.133333    85.6955667   FALSE FALSE
## yaw_forearm          16.106509    12.9067482   FALSE FALSE
## total_accel_belt      1.077358     0.2111087   FALSE FALSE
## total_accel_arm       1.079872     0.4804542   FALSE FALSE
## total_accel_dumbbell  1.074816     0.3057436   FALSE FALSE
## total_accel_forearm   1.088664     0.5022931   FALSE FALSE
```
Because the variables differ in their magnitude, standardization will be used.

## Prediction Analysis

Following the authors of the paper mentioned above, a random forest approach is applied, 
including cross validation. 

```r
set.seed(456)
cv <- trainControl(method = "cv", number = 5)
modFit <- train(classe ~ ., data = TrainingSet, method = "rf", preProcess = 
                        c("center", "scale"), trControl = cv)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
print(modFit)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10988, 10991, 10990, 10989, 10990 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9906821  0.9882121
##   27    0.9913376  0.9890422
##   52    0.9873338  0.9839767
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```
From the printout, we can see that the model produces very accurate predictions for the way 
the exercises were executed (variable classe). The accuracy is 99.1 percent. However, the 
computation was time consuming. Therefore, two other prediction models are tested. First, 
linear discriminant analysis is applied.

```r
set.seed(789)
modFit2 <- train(classe ~ ., data = TrainingSet, method = "lda", preProcess = 
                         c("center", "scale"), trControl = cv)
```

```
## Loading required package: MASS
```

```r
print(modFit2)
```

```
## Linear Discriminant Analysis 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10991, 10989, 10989, 10990 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.7021929  0.6231017
```
With an accuracy of 70.2 percent, this model produces less accurate predictions 
than the random forest method. Second, the quadratic discriminant analysis is applied.

```r
set.seed(147)
modFit3 <- train(classe ~ ., data = TrainingSet, method = "qda", preProcess = 
                         c("center", "scale"), trControl = cv)
print(modFit3)
```

```
## Quadratic Discriminant Analysis 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10990, 10990, 10990, 10989 
## Resampling results:
## 
##   Accuracy  Kappa    
##   0.896484  0.8692189
```
With an accuracy of 86.9 precent, this model produces less accurate predictions than the
random forest method. 

### Out of sample error

Because of its high accuracy, the random forest method is used to predict the class
of exercises in the testing set (subsample of training set) in order to get the out of sample error.

```r
prediction <- predict(modFit,newdata=TestingSet)
confusionMatrix(prediction,TestingSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    8    0    0    0
##          B    1 1128    8    0    0
##          C    0    3 1015   18    2
##          D    0    0    3  945    1
##          E    0    0    0    1 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9924          
##                  95% CI : (0.9898, 0.9944)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9903          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9903   0.9893   0.9803   0.9972
## Specificity            0.9981   0.9981   0.9953   0.9992   0.9998
## Pos Pred Value         0.9952   0.9921   0.9778   0.9958   0.9991
## Neg Pred Value         0.9998   0.9977   0.9977   0.9962   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1917   0.1725   0.1606   0.1833
## Detection Prevalence   0.2856   0.1932   0.1764   0.1613   0.1835
## Balanced Accuracy      0.9988   0.9942   0.9923   0.9897   0.9985
```
Here, the prediction accuracy is 99.24 percent.

### Prediction

Finally, predictions for the obervations in the test set are produced.

```r
predictionfinal <- predict(modFit,newdata=testing)
predictionfinal
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
print(rbind(testing$X, as.character(predictionfinal)))
```

```
##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13]
## [1,] "1"  "2"  "3"  "4"  "5"  "6"  "7"  "8"  "9"  "10"  "11"  "12"  "13" 
## [2,] "B"  "A"  "B"  "A"  "A"  "E"  "D"  "B"  "A"  "A"   "B"   "C"   "B"  
##      [,14] [,15] [,16] [,17] [,18] [,19] [,20]
## [1,] "14"  "15"  "16"  "17"  "18"  "19"  "20" 
## [2,] "A"   "E"   "E"   "A"   "B"   "B"   "B"
```









