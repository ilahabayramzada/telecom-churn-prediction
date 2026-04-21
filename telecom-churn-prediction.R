library(tidyverse)
library(caret)
library(scales)
library(dplyr)
library(inspectdf)

data <- read.csv("telecom.csv")
data %>% glimpse()

# Cleaning and preprocessing the data ----

# NA dəyərləri yoxlamaq
inspect_na(data)

# NA-ları silmək
data <- data %>% drop_na()

# Churn və SeniorCitizen sütunlarını factor-a çevirmək
data$Churn <- as.factor(data$Churn)
data$SeniorCitizen <- as.factor(data$SeniorCitizen)

# One-hot encoding
dummies <- dummyVars(" ~ .", data = data)
data_encoded <- data.frame(predict(dummies, newdata = data))

# Numerik sütunları seçmək
numeric_cols <- sapply(data_encoded, is.numeric)

# Min-Max scaling
data_encoded[numeric_cols] <- lapply(data_encoded[numeric_cols], function(x) rescale(x, to = c(0, 1)))

#Outlier-larin temizlenmesi
num_vars <- names(data_encoded)[numeric_cols]

for (var in num_vars) {
  out_vals <- boxplot(data_encoded[[var]], plot = FALSE)$out
  if (length(out_vals) > 0) {
    q1 <- quantile(data_encoded[[var]], 0.25)
    q3 <- quantile(data_encoded[[var]], 0.75)
    iqr <- IQR(data_encoded[[var]])
    lower <- q1 - 1.5 * iqr
    upper <- q3 + 1.5 * iqr
    data_encoded[[var]][data_encoded[[var]] < lower] <- lower
    data_encoded[[var]][data_encoded[[var]] > upper] <- upper
  }
}


#Exploratory Data Analysis(EDA)----

library(tidyverse)
library(inspectdf)
library(naniar)
library(correlationfunnel)
library(explore)
library(SmartEDA)

#inspectdf ilə struktur və korrelyasiya analizi
data %>% inspect_na() %>% show_plot()
data %>% inspect_cor() %>% show_plot()
data %>% inspect_num() %>% show_plot()
data %>% inspect_cat() %>% show_plot()


#explore ilə dəyişənlərin təsviri və vizual təhlili
data %>% describe()
data %>% describe_tbl()

data %>% explore(Churn)
data %>% explore(MonthlyCharges)
data %>% explore(MonthlyCharges, target = Churn)
data %>% explore(tenure, MonthlyCharges, target = Churn)

#Feature Engineering----

#The most relevant features
library(car)
target <- "Churn"

df<-data

numeric_features <- df %>% 
  select(-all_of(target)) %>% 
  select(where(function(x) is.numeric(x))) %>% 
  names()

vif_df <- df[, numeric_features]

# Constant sütunları çıxar
vif_df <- vif_df[, sapply(vif_df, function(x) length(unique(x)) > 1)]

# VIF hesabla
# lm üçün target lazım olduğu üçün hər hansı numeric sütun istifadə oluna bilər, amma nəticəyə təsir etmir
vif_formula <- as.formula(paste(numeric_features[1], "~ ."))  # birinci numeric sütun target kimi
vif_model <- lm(vif_formula, data = vif_df)

vif_values <- car::vif(vif_model)

vif_df_final <- data.frame(
  feature = names(vif_values),
  VIF = as.numeric(vif_values)
) %>% arrange(desc(VIF))

vif_df_final

while (vif_df_final$VIF %>% .[1] >= 2) {
  features <- vif_df_final[-1,"feature"]
  
  vif_df_final <- sapply(features, function(target) {
    formula <- as.formula(paste(target, "~ ."))
    model <- lm(formula, data = df[, features])
    1 / (1 - summary(model)$r.squared)
  })
  
  vif_df_final <- vif_df_final %>% 
    as.data.frame() %>% 
    rownames_to_column() %>% 
    rename(VIF = ".",
           feature = rowname) %>% 
    arrange(desc(VIF))
}

# "Weight Of Evidence" 
library(scorecard)

target <- "Churn"

df$Churn <- df$Churn %>%
  recode("'Yes' = 1; 'No' = 0")%>%
  as_factor()

df$Churn %>% table() %>% prop.table() %>% round(2)

# IV (information values) 
iv <- df %>% 
  iv(y = target, positive = "1") %>% 
  as_tibble() %>%
  mutate(info_value = round(info_value, 3)) %>%
  arrange(desc(info_value))

# IV dəyəri 0.02'dən kiçik olan dəyişənləri çıxarmaq
ivars <- iv %>% 
  filter(info_value > 0.02) %>% 
  pull(variable) 

df.iv <- df %>% select(all_of(target),all_of(ivars))

df.iv %>% dim()

# Datanın bölünməsi
dt_list <- df.iv %>% 
  split_df(target, ratio = 0.8, seed = 123)

# woe binning 

bins <- dt_list$train %>% 
  woebin(y = "Churn", positive = "1")

bins$OnlineBackup %>% as_tibble()
bins$OnlineBackup %>% woebin_plot()

train_woe <- dt_list$train %>% woebin_ply(bins) 
test_woe <- dt_list$test %>% woebin_ply(bins)

names <- train_woe %>% 
  names() %>% 
  str_replace_all("_woe","")  

names(train_woe) <- names
names(test_woe) <- names

#Logistic Regression(GLM)----

features <- setdiff(names(train_woe), target)

train <- train_woe %>% select(all_of(c(target, features)))
test  <- test_woe %>% select(all_of(c(target, features)))

f <- as.formula(paste(target, "~", paste(features, collapse = " + ")))

glm_model <- glm(f, data = train, family = "binomial")
summary(glm_model)

# Proqnozlar
pred_prob <- glm_model %>% 
  predict(newdata = test)

library(MLmetrics)
library(yardstick)  

actual <- ifelse(test[[target]] == 1, 1, 0)

# Optimal threshold
thresholds <- seq(0, 1, 0.01)
f1_scores <- sapply(thresholds, function(thr){
  preds_class <- ifelse(pred_prob > thr, 1, 0)
  F1_Score(y_true = actual, y_pred = preds_class)
})

opt_thresh <- thresholds[which.max(f1_scores)]
pred_class <- ifelse(pred_prob > opt_thresh, 1, 0)

cm <- table(Actual = actual, Predicted = pred_class)

# Metriklər
TP <- cm[2,2]
FP <- cm[1,2]
FN <- cm[2,1]

accuracy  <- sum(diag(cm)) / sum(cm)
precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
f1        <- 2 * precision * recall / (precision + recall)

library(pROC)
auc <- roc(actual, pred_prob)$auc

glm_metrics <- list(
  confusion_matrix = cm,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1,
  auc = auc
)

glm_metrics

# Datanın bölünməsi - Splitting the data ----

library(tidymodels)
library(discrim)
library(bonsai)
library(rsample)

data$Churn <- data$Churn %>%
  car::recode("'Yes' = 1; 'No' = 0") %>%
  as_factor()

data$Churn %>% table() %>% prop.table() %>% round(2)


target <- "Churn"
exclude <- c("customerID")

data_split <- data %>% 
  select(-all_of(exclude)) %>% 
  initial_split(prop = 0.8, strata = target)
train <- training(data_split)
test  <- testing(data_split)

# Alqoritmalar ----

library(yardstick)

form <- as.formula(paste(target, "~ ."))

metrics_cls <- yardstick::metric_set(
  yardstick::accuracy,
  yardstick::f_meas,
  yardstick::precision,
  yardstick::recall
)

# Decision Tree
tree_spec <- decision_tree(
  mode = "classification",
  tree_depth = 15,
  min_n = 2) %>%
  set_engine("rpart")

tree_wf <- workflow() %>% 
  add_model(tree_spec) %>% 
  add_formula(form)

tree_fit <- tree_wf %>% 
  fit(train)

tree_pred <- predict(tree_fit, test, type = "prob") %>%
  bind_cols(predict(tree_fit, test, type = "class")) %>%
  mutate(truth = test[[target]])

auc <- roc_auc_vec(
  truth = tree_pred$truth, 
  estimate = as.numeric(as.character(tree_pred$.pred_0))
)

tree_metrics <- tree_pred %>% 
  metrics_cls(truth = truth, estimate = .pred_class) %>% 
  bind_rows(tibble(
    .metric = "auc",
    .estimate = auc
  ))

# Random Forest (Bagging)

rf_spec <- rand_forest(
  mode = "classification",
  trees = 500,
  min_n = 5) %>%
  set_engine("ranger")

rf_wf <- workflow() %>% 
  add_model(rf_spec) %>% 
  add_formula(form)

rf_fit <- rf_wf %>% 
  fit(train)

rf_pred <- predict(rf_fit, test, type = "prob") %>%
  bind_cols(predict(rf_fit, test, type = "class")) %>%
  mutate(truth = test[[target]])

auc <- roc_auc_vec(
  truth = rf_pred$truth, 
  estimate = as.numeric(as.character(rf_pred$.pred_0))
)

rf_metrics <- rf_pred %>% 
  metrics_cls(truth = truth, estimate = .pred_class) %>% 
  bind_rows(tibble(
    .metric = "auc",
    .estimate = auc
  ))

# XGBoost (Boosting)
xgb_spec <- boost_tree(
  mode = "classification",
  trees = 1000,
  learn_rate = 0.05,
  tree_depth = 6,
  min_n = 5) %>%
  set_engine("xgboost")

xgb_wf <- workflow() %>% 
  add_model(xgb_spec) %>% 
  add_formula(form)

xgb_fit <- xgb_wf %>% 
  fit(train)

xgb_pred <- predict(xgb_fit, test, type = "prob") %>%
  bind_cols(predict(xgb_fit, test, type = "class")) %>%
  mutate(truth = test[[target]])

auc <- roc_auc_vec(
  truth = xgb_pred$truth, 
  estimate = as.numeric(as.character(xgb_pred$.pred_0))
)

xgb_metrics <- xgb_pred %>% 
  metrics_cls(truth = truth, estimate = .pred_class) %>% 
  bind_rows(tibble(
    .metric = "auc",
    .estimate = auc
  ))


# Ən yaxşı model seçimi ----

bind_rows(
  mutate(tree_metrics, model = "Decision Tree"),
  mutate(rf_metrics, model = "Random Forest"),
  mutate(xgb_metrics, model = "XGBoost")
) %>% 
  select(-.estimator) %>% 
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(accuracy))

