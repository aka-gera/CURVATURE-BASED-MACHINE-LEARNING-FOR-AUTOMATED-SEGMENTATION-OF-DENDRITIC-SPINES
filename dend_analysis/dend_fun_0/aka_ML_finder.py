import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

from scipy import stats




class aka_ML_analysis:
  def __init__(self, tcouleur='plotly_dark', bcouleur='navy', fcouleur='white', fsize=20):
      self.tcouleur = tcouleur
      self.bcouleur = bcouleur
      self.fcouleur = fcouleur
      self.fsize = fsize
      self.update_layout_parameter = dict(
                                      barmode='overlay',
                                      font=dict(color=fcouleur,size=fsize),
                                      title_x=0.5,
                                      title_y=0.9,
                                      template=self.tcouleur
                                      )
      self.update_axes = dict(
                          title_font = {"size": 14},
                          title_standoff = 25
                          )

  def regression_summary(self, X_train, y_train, y_pred, params, feat_name, reg = 1):

      nn, kk = X_train.shape
      # Degrees of freedom.
      degf = float(nn-kk-1)
      dff = pd.DataFrame(X_train)
      dff.columns = feat_name
      if reg == 1:
        df_tmp = pd.DataFrame({"Constant":np.ones(nn)}).join(dff)
      else:
        df_tmp = dff

      MSE = (sum((y_train-y_pred)**2))/degf
      

      var_beta = MSE*(np.linalg.inv(np.dot(df_tmp.T,df_tmp)).diagonal())
      sd_beta = np.sqrt(var_beta) 
      ts_beta = params/ sd_beta

      p_values =[2*(1-stats.t.cdf(np.abs(its),degf)) for its in ts_beta]

      sd_beta = np.round(sd_beta,3)
      ts_beta = np.round(ts_beta,3)
      p_values = np.round(p_values,4)
      params = np.round(params,4)



      alpha = 0.05  # Significance level (1 - confidence level)
      critical_value = stats.t.ppf(1-alpha/2, degf)

      # Compute the confidence intervals for the coefficients
      low_bound = params - critical_value * sd_beta
      up_bound = params + critical_value * sd_beta

      low_bound = np.round(low_bound,3)
      up_bound = np.round(up_bound,3)

      df_res = pd.DataFrame()
      df_res["features"],df_res["Coef"],df_res["str err"],df_res["t values"],df_res["P > |t|"],df_res["[0.025  "],df_res["   0.975]"] = [df_tmp.columns,params,sd_beta,ts_beta,p_values,low_bound,up_bound]
  
      return df_res



  def plot_regression_summary(self, X_train, y_train, y_pred, params, cmLabel, lab=1): 
      # Generate classification report
      df_res = self.regression_summary(X_train, y_train, y_pred, params, cmLabel)
      colss =  [mn for mn in df_res.columns[1:]] 
  
      df_name = [mn for mn in df_res[df_res.columns[0]]] 
      # df_name.reverse() 

      cm = df_res[df_res.columns[1:]].fillna(0).values 

      if lab == 1:
          fig = ff.create_annotated_heatmap(cm,
                                          x=colss,
                                          y=df_name,
                                          # annotation_text=cm.round(3),
                                          colorscale='Viridis',)
      else:
          fig = ff.create_annotated_heatmap(cm,
                                          x=colss,
                                          colorscale='Viridis')
          fig.update_yaxes(title_text='y', showticklabels=False)

      fig.update_layout(title='Regression Summary')
      fig.update_layout(**self.update_layout_parameter) 
      fig.update_xaxes(**self.update_axes)
      fig.update_yaxes(**self.update_axes) 

      return fig


  def plot_metric_report(self,df, lab=1):
      x_label = df.columns
      y_label = df.index

      cm = df.values.round(3)

      fig = go.Figure(data=go.Heatmap(
          z=cm,
          x=x_label,
          y=y_label,
          colorscale='Viridis', 
          hoverongaps=False))

      fig.update_layout(
          xaxis_title='ML algorithm',
          yaxis_title='Metric',
          title='Metric Report', 
          font=dict(size=12)
      )

      if lab == 1:
          for i in range(len(y_label)):
              for j in range(len(x_label)):
                  fig.add_annotation(
                      x=x_label[j],
                      y=y_label[i],
                      text=str(cm[i, j]),
                      showarrow=False,
                      font=dict(color='black', size=12),
                      xanchor='center',
                      yanchor='middle'
                  )
      fig.update_layout(**self.update_layout_parameter)
      fig.update_xaxes(**self.update_axes)
      fig.update_yaxes(**self.update_axes)

      return fig



    
  def plot_confusion_matrix(self,y,y_predict,cmLabel,lab=1):
    cm = confusion_matrix(y, y_predict)
    if lab == 1:
        fig = ff.create_annotated_heatmap(cm,
                                        x=cmLabel[:cm.shape[1]],
                                        y=cmLabel[:cm.shape[1]],
                                        colorscale='Viridis',showscale=True)
        fig.update_xaxes(
                title_text='Predicted labels', 
                side='bottom')
        fig.update_yaxes(title_text = 'True labels')
    else:
        annotation_text = [['' for _ in range(cm.shape[1])] for _ in range(cm.shape[0])]
        fig = ff.create_annotated_heatmap(cm,
                                        x=cmLabel[:cm.shape[1]],
                                        y=cmLabel[:cm.shape[1]],
                                        colorscale='Viridis',
                                        annotation_text=annotation_text,
                                        showscale=True)
        fig.update_xaxes(
                title_text='Prediction', 
                side='bottom')
        fig.update_xaxes( showticklabels=True )
        fig.update_yaxes(title_text = 'True Solution')
        fig.update_yaxes(showticklabels=True )

    fig.update_layout(title='Confusion Matrix') 
    fig.update_layout(**self.update_layout_parameter)
    fig.update_xaxes(**self.update_axes)
    fig.update_yaxes(**self.update_axes)

    return fig

  def plot_classification_report_default(self,y, y_predict,cmLabel,lab=1): 
    
    report_str = classification_report(y, y_predict)
    report_lines = report_str.split('\n')

    # Remove empty lines
    report_lines = [line for line in report_lines if line.strip()]
    data = [line.split() for line in report_lines[1:]]
    colss = ['feature', 'precision',   'recall',  'f1-score',   'support', 'n1one']

    # Convert to a DataFrame
    report_df = pd.DataFrame(data, columns = colss )
    report_df = report_df[report_df.columns[:-1]] 
    cm = report_df.iloc[:,1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    colss1 = [  'precision',   'recall',  'f1-score',   'support']

    if lab == 1:
        fig = ff.create_annotated_heatmap(cm,
                                            x = colss1,
                                            y = cmLabel[:cm.shape[0]],
                                            colorscale='Viridis' ) 
    else:
        cmm =  cm[:,:-1]
        annotation_text = [['' for _ in range(cmm.shape[1])] for _ in range(cmm.shape[0])]
        fig = ff.create_annotated_heatmap(cmm,
                                            x = colss1[:-1],
                                            colorscale='Viridis',
                                            showscale=True,
                                            annotation_text=annotation_text )
        fig.update_yaxes(
                title_text = 'y', 
                showticklabels=False  
                )
    fig.update_layout(title='Classification Report')
    fig.update_layout(**self.update_layout_parameter) 
    fig.update_xaxes(**self.update_axes)
    fig.update_yaxes(**self.update_axes) 

    return fig

  def plot_classification_report(self, y, y_predict, cmLabel, lab=1): 
    # Generate classification report
    report_str = classification_report(y, y_predict, target_names=cmLabel, output_dict=True) 
    colss = ['precision', 'recall', 'f1-score', 'support']

    # Convert to a DataFrame
    report_df = pd.DataFrame(report_str)
    report_df = report_df.drop(report_df.columns[-3],axis=1)
    df_name = [mn for mn in report_df.columns] 
    df_name.reverse() 

    cm = report_df.apply(pd.to_numeric, errors='coerce').fillna(0).values.T 
    if lab == 1:
        fig = ff.create_annotated_heatmap(cm,
                                        x=colss,
                                        y=df_name,
                                        annotation_text=cm.round(3),
                                        colorscale='Viridis')
    else:
        fig = ff.create_annotated_heatmap(cm,
                                        x=colss,
                                        colorscale='Viridis')
        fig.update_yaxes(title_text='y', showticklabels=False)

    fig.update_layout(title='Classification Report')
    fig.update_layout(**self.update_layout_parameter) 
    fig.update_xaxes(**self.update_axes)
    fig.update_yaxes(**self.update_axes) 

    return fig


  def plot_important_features(self, model,df,max_num_feat=10):
    import_features = model.feature_importances_
    cols = df.columns[:-1]
    max_num_feat = min(len(cols),max_num_feat)

    importance_cols = zip(import_features, cols)
    
    sorted_importance_cols = sorted(importance_cols, key=lambda x: abs(x[0]), reverse=True)

    sorted_import_features, sorted_cols = zip(*sorted_importance_cols)

    data = pd.DataFrame({'Weight': sorted_import_features[:max_num_feat], 'Feature Name': sorted_cols[:max_num_feat]})

    fig = px.bar(data, x='Weight', y='Feature Name', orientation='h', title=f'Top {max_num_feat} Importance Features by Weight')


    fig.update_layout(**self.update_layout_parameter) 
    fig.update_xaxes(**self.update_axes)
    return fig




from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error, max_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV 


class aka_regression:
  def __init__(self):
      self.regressors = {
          'Linear Regression': LinearRegression(),
          'SGD Regression': SGDRegressor(),
          'Ridge Regression': Ridge(),
          'Decision Tree Regression': DecisionTreeRegressor(),
          'Random Forest Regression': RandomForestRegressor(),
          'AdaBoost Regression': AdaBoostRegressor(),
          'Gradient Boost Regression': GradientBoostingRegressor(),
          'XGBoost Regression': XGBRegressor(),
          'Bagging Regression': BaggingRegressor(),
          'Hist Gradient Boosting Regression': HistGradientBoostingRegressor(),
          'Extra Tree Regression': ExtraTreesRegressor(),
          'Cat Boost Regression': CatBoostRegressor(verbose=0)
      }

      self.metrics = {
          'MAE': mean_absolute_error,
          'MSE': mean_squared_error,
          'MSLE': mean_squared_log_error,
          'Median Error': median_absolute_error,
          'MAPE': mean_absolute_percentage_error,
          'Max Error': max_error,
          'R2 Score': r2_score
      }

  def train_and_find_best_regressor(self,X_train, y_train, X_test, y_test):
      parameters = {
          # 'learning_rate': [0.1, 0.01, 0.05],
          # 'max_depth': [3, 5, 7],
          # 'n_estimators': [70, 100, 200,500]
      }  
      best_algorithms = {}
      metric_algorithms = {}
      clf_algorithms = {}

      for regressor_name, clf in self.regressors.items():
          clf = GridSearchCV(estimator=clf, param_grid=parameters, refit="recall", cv=3) 
          clf.fit(X_train, y_train)
          clf_algorithms[regressor_name] = clf

      for metric_name, metric_func in self.metrics.items():

          best_error = float('inf')
          best_regressor = None

          for regressor_name, clf in clf_algorithms.items():
              predictions = clf.predict(X_test)
              error = metric_func(y_test, predictions)

              metric_algorithms.setdefault(regressor_name, {})[metric_name] = error

              if metric_name == 'R2 Score':  # R2 Score needs to be maximized
                  error = abs(error - 1)  # Convert to a form that can be minimized
              else:
                  error = abs(error)  # Other metrics should be minimized

              if error < best_error:
                  best_error = error
                  best_regressor = regressor_name 

          best_algorithms[metric_name] = best_regressor
      ml_vals = [mm for mm in best_algorithms.values()]
      most_common_metric = max(ml_vals, key=ml_vals.count) 
      df_metric_algorithms = pd.DataFrame(metric_algorithms)

      return clf_algorithms[most_common_metric], df_metric_algorithms, clf_algorithms
 


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class aka_classification:
    def __init__(self,classifiers=None):
        if classifiers is None:
            classifiers = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                # 'SVM (Linear Kernel)': SVC(kernel='linear'),
                # 'SVM (RBF Kernel)': SVC(kernel='rbf'),
                # 'SVM (Poly Kernel)': SVC(kernel='poly'),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                # 'Naive Bayes': GaussianNB(),
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'Random Forest Classifier': RandomForestClassifier(),
                # 'AdaBoost Classifier': AdaBoostClassifier(),
                # 'Gradient Boost Classifier': GradientBoostingClassifier(),
                'XGBoost Classifier': XGBClassifier(),
                # 'Bagging Classifier': BaggingClassifier(),
                # 'Extra Tree Classifier': ExtraTreesClassifier(),
                # 'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
                # 'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
                # 'Cat Boost Classifier': CatBoostClassifier(verbose=0)
            }
        self.classifiers=classifiers
        self.metrics = {
            'Accuracy': accuracy_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'F1 Score': f1_score
        }


    def train(self, X_train, y_train,  cv=3, 
        clf_algorithms = {},
        parameters = {
            # 'learning_rate': [0.1, 0.01, 0.05],
            # 'max_depth': [3, 5, 7],
            # 'n_estimators': [70, 100, 200,500]
        } , ):

        for classifier_name, clf in self.classifiers.items():
            clf = GridSearchCV(estimator=clf, param_grid=parameters, refit="recall", cv=cv) 
            clf.fit(X_train, y_train)
            clf_algorithms[classifier_name] = clf
        return clf_algorithms


    def train_and_find_best_classifier(self, X_test, y_test,  
                                        best_algorithms = {},
                                        metric_algorithms = {},
                                        clf_algorithms = {}, 
                                        ):
        
     
        predictions = {classifier_name:clf.predict(X_test) for classifier_name, clf in clf_algorithms.items()}

        for metric_name, metric_func in self.metrics.items():
            best_error = 0
            best_classifier = None
            for classifier_name,pred in predictions.items():
                error = metric_func(y_test, pred)

                metric_algorithms.setdefault(classifier_name, {})[metric_name] = error

                error = abs(error)

                if error > best_error:
                    best_error = error
                    best_classifier = classifier_name  

            best_algorithms[metric_name] = best_classifier 

        ml_vals = [mm for mm in best_algorithms.values()]
        most_common_metric = max(ml_vals, key=ml_vals.count) 
        df_metric_algorithms = pd.DataFrame(metric_algorithms)

        return clf_algorithms[most_common_metric], df_metric_algorithms 

