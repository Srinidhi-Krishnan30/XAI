# %%
import numpy as np
import pandas as pd

from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib
import warnings

# %%
# df = pd.read_csv('Battery_RUL.csv')
df = pd.read_csv("Battery_RUL.csv")

# %%
df.head()

# %%
df["Battery ID"] = 0
batteries = []
ID = 1
for rul in df["RUL"]:
    batteries.append(ID)
    if rul == 0:
        ID += 1
        continue
df["Battery ID"] = batteries

# %%
df.head()

# %%
sensor_list = df.columns[1:-2]
sensor_list

# %%
df.info()

# %%
df.describe(include="all").T

# %%
train_battery_ids = []
test_battery_ids = []
battery_ids = df["Battery ID"].unique()

for i in battery_ids:
    if i < 9:
        train_battery_ids.append(i)
    else:
        test_battery_ids.append(i)
df_train = df[df["Battery ID"].isin(train_battery_ids)]
df_test = df[df["Battery ID"].isin(test_battery_ids)]

# %%
plt.figure(figsize=(10, 10))
threshold = 0.90
sns.set_style("whitegrid", {"axes.facecolor": ".0"})
df_cluster2 = df_train.corr()
mask = df_cluster2.where((abs(df_cluster2) >= threshold)).isna()
plot_kws = {"s": 1}
sns.heatmap(
    df_cluster2,
    cmap="RdYlBu",
    annot=True,
    mask=mask,
    linewidths=0.2,
    linecolor="lightgrey",
).set_facecolor("white")
plt.savefig('images/correlation_heatmap.png')
plt.close()

# %%
sens_const_values = []
for feature in list(sensor_list):
    try:
        if df_train[feature].min() == df_train[feature].max():
            sens_const_values.append(feature)
    except:
        pass

print(sens_const_values)
df_train.drop(sens_const_values, axis=1, inplace=True)
df_test.drop(sens_const_values, axis=1, inplace=True)

# %%
# corr_features = ['sensor_9']

cor_matrix = df_train[sensor_list].corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
corr_features = [
    column for column in upper_tri.columns if any(upper_tri[column] > 0.95)
]
print(corr_features)
df_train.drop(corr_features, axis=1, inplace=True)
df_test.drop(corr_features, axis=1, inplace=True)

# %%
list(df_train)

# %%
df_train.head()

# %%
features = list(df_train.columns)

# %%
for feature in features:
    print(feature + " - " + str(len(df_train[df_train[feature].isna()])))

# %%
print(plt.style.available)

# %%
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = 8, 25
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8
plt.rcParams["lines.linewidth"] = 0.5
plot_items = list(df_train.columns)[1:-2]
fig, ax = plt.subplots(len(plot_items), sharex=True)
ax[0].invert_xaxis()

batteries = list(df_train["Battery ID"].unique())
batteries_test = list(df_test["Battery ID"].unique())

for battery in batteries:
    for i, item in enumerate(plot_items):
        f = sns.lineplot(
            data=df_train[df_train["Battery ID"] == battery],
            x="RUL",
            y=item,
            color="steelblue",
            ax=ax[i],
        )
plt.savefig('images/line_plots.png')
plt.close()

# %%
Selected_Features = []
import statsmodels.api as sm


def backward_regression(X, y, initial_list=[], threshold_out=0.05, verbose=True):
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"worst_feature : {worst_feature}, {worst_pval} ")
        if not changed:
            break
    Selected_Features.append(included)
    print(f"\nSelected Features:\n{Selected_Features[0]}")


# Application of the backward regression function on our training data
X = df_train.iloc[:, 1:-2]
y = df_train.iloc[:, -1]
backward_regression(X, y)

# %%
feature_names = Selected_Features[0]

# %%
import time

model_performance = pd.DataFrame(columns=["r-Squared", "RMSE", "total time"])

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, accuracy_score

import sklearn
from sklearn.metrics import mean_squared_error, r2_score

model_performance = pd.DataFrame(
    columns=["R2", "RMSE", "time to train", "time to predict", "total time"]
)


def R_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# %%
X_train = df_train[feature_names]
y_train = df_train["RUL"]

X_test = df_test[feature_names]
y_test = df_test["RUL"]

# %%
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# %%
from sklearn.ensemble import RandomForestRegressor

start = time.time()
model = RandomForestRegressor(
    n_jobs=-1,
    n_estimators=100,
    min_samples_leaf=1,
    max_features="sqrt",
    # min_samples_split=2,
    bootstrap=True,
    criterion="squared_error",
).fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test)  # These are the predictions from the test data.
end_predict = time.time()

model_performance.loc["Random Forest"] = [
    model.score(X_test, y_test),
    mean_squared_error(y_test, y_predictions, squared=False),
    end_train - start,
    end_predict - end_train,
    end_predict - start,
]

print("R-squared error: " + "{:.2%}".format(model.score(X_test, y_test)))
print(
    "Root Mean Squared Error: "
    + "{:.2f}".format(mean_squared_error(y_test, y_predictions, squared=False))
)

# %%
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = 5, 5

fig, ax = plt.subplots()
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
g = sns.scatterplot(
    x=y_test, y=y_predictions, s=20, alpha=0.6, linewidth=1, edgecolor="black", ax=ax
)
plt.savefig('images/scatterPlot.png')
plt.close()
f = sns.lineplot(
    x=[min(y_test), max(y_test)],
    y=[min(y_test), max(y_test)],
    linewidth=4,
    color="gray",
    ax=ax,
)

plt.annotate(
    text=(
        "R-squared error: "
        + "{:.2%}".format(model.score(X_test, y_test))
        + "\n"
        + "Root Mean Squared Error: "
        + "{:.2f}".format(mean_squared_error(y_test, y_predictions, squared=False))
    ),
    xy=(0, 800),
    size="medium",
)

xlabels = ["{:,.0f}".format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ["{:,.0f}".format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()
plt.savefig('images/linePlot.png')
plt.close()

# %%
plt.rcParams["figure.figsize"] = 5, 10
sns.set_style("white")
feat_importances = pd.Series(model.feature_importances_, index=feature_names)
feat_importances = feat_importances.groupby(level=0).mean()
feat_importances.nlargest(20).plot(kind="barh").invert_yaxis()
sns.despine()
plt.savefig('images/features.png')
plt.close()

# %%
df_test.head()

# %%
df_test["RUL predicted"] = y_predictions

# %%
batteries = list(df_train["Battery ID"].unique())
batteries_test = list(df_test["Battery ID"].unique())

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = 8, 25
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8
plt.rcParams["lines.linewidth"] = 1.5
fig, ax = plt.subplots(len(batteries_test), sharex=True)

for i, battery in enumerate(batteries_test):
    f = sns.lineplot(
        data=df_test[df_test["Battery ID"] == battery],
        x="Cycle_Index",
        y="RUL",
        color="dimgray",
        ax=ax[i],
        label="Actual",
    )
    g = sns.lineplot(
        data=df_test[df_test["Battery ID"] == battery],
        x="Cycle_Index",
        y="RUL predicted",
        color="steelblue",
        ax=ax[i],
        label="Predicted",
    )
    ax[i].legend = True
plt.savefig('images/rul_prediction_per_battery.png')
plt.close()

# %%
model_performance.style.background_gradient(cmap="RdYlBu_r").format(
    {
        "R2": "{:.2%}",
        "RMSE": "{:.2f}",
        "time to train": "{:.3f}",
        "time to predict": "{:.3f}",
        "total time": "{:.3f}",
    }
)

# %%
from lime.lime_tabular import LimeTabularExplainer

# %%
print(X.shape)
print(X_train.shape)

# %%
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# %%
feature_names = list(X_train_df.columns)
# Fit the Explainer on the training data set using the LimeTabularExplainer
explainer = LimeTabularExplainer(
    training_data=X_train_df.values,  # Use .values to get NumPy array from DataFrame
    feature_names=X_train_df.columns.tolist(),  # List of feature names
    mode="regression",
)

# %%
i = np.random.randint(0, X_test_df.shape[0])
exp = explainer.explain_instance(
    X_test_df.iloc[i].values, model.predict, num_features=10
)
exp.show_in_notebook(show_table=True)

# %% [markdown]
# # Inference
#
# For the random data point passed into the model the predicted value of RUL is 643.33. The range of prediction of values was inferred to be between 16.82 and 1107.76 cycles. The feature 0 bearing a value of 0.71 has a positive impact of 144.35 on the result predicted. Feature 1 on the other hand has a negative impact of 119.18. The increase of contribution of this feature lowers the predicted RUL.In the given instance a value of 0.61 to this feature lowers the number of cycles by 119.18. The index of the feature is determined by its position in the dataframe. The feature 1 could possibly be discharge Time and feature 0 could be cycle time as this inference is supported by the heatmap drawn earlier with values 0.92 and -1 respectively
#

# %%
from sklearn.inspection import PartialDependenceDisplay

# %%
feature_index = 0  # Index of the feature you want to analyze (e.g., the first feature)

# Create ICE plot
fig, ax = plt.subplots(figsize=(10, 6))
display = PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    [feature_index],
    kind="both",  # 'both' for PDP and ICE, or 'individual' for ICE only
    grid_resolution=50,
    ax=ax,
    ice_lines_kw={"alpha": 0.3, "color": "gray"},
    pd_line_kw={"color": "red"},
)
plt.suptitle("Individual Conditional Expectation (ICE) Plot")
plt.savefig('images/ICEPlot.png')
plt.close()

# %% [markdown]
# # Inference from the plot
#
# The grey lines captures the change in prediction of a single data instance as the value of the specific feature is varied keeping other features constant. The upward movement of lines indicates that the feature has a positive correlation on the target variable. At lower levels the variance between the data points is high indicating that the impact of the specific feature keeps changing for different data points. The red line indicates the average effect of the feature on the RUL. As this is not a straight line it indicates a non linear dependence between the feature and the target variable
#

# %%
import dice_ml
from dice_ml import Dice
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("./Battery_RUL.csv")

# Define feature names
feature_names = [
    "Cycle_Index",
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
]

# Sanitize the column names
sanitized_feature_names = [
    col.replace(" ", "_")
    .replace(".", "")
    .replace("(", "")
    .replace(")", "")
    .replace("-", "_")
    for col in feature_names
]
df.columns = [
    col.replace(" ", "_")
    .replace(".", "")
    .replace("(", "")
    .replace(")", "")
    .replace("-", "_")
    for col in df.columns
]

# Split the data into features (X) and target (y)
X = df[sanitized_feature_names]
y = df["RUL"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Define a custom prediction function
def custom_predict(features):
    features_df = pd.DataFrame(features, columns=sanitized_feature_names)
    predictions = model.predict(features_df)
    return predictions.tolist()


# Combine your test features and labels into a DataFrame
test_data = X_test.copy()
test_data["RUL"] = y_test.values

# %%
# Initialize DiCE: First, create a data object for DiCE
data_dice = dice_ml.Data(
    dataframe=df, continuous_features=sanitized_feature_names, outcome_name="RUL"
)

# Create a model object for DiCE
model_dice = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")

# Initialize DiCE explainer
dice_exp = Dice(data_dice, model_dice)

# Select an instance for which you want to generate counterfactuals (e.g., the first test instance)
query_instance = X_test.iloc[0:1]

# Generate counterfactuals
dice_cf = dice_exp.generate_counterfactuals(
    query_instance, total_CFs=5, desired_range=[y.min(), y.max()]
)

# Visualize the generated counterfactuals
dice_cf.visualize_as_dataframe()

# %% [markdown]
# # Counterfactual Explanation:
#
# The counterfactual examples generated by DiCE provide alternative feature values that lead to slightly different predicted RUL values. Here’s a breakdown of the counterfactual instances:
#
# 1. Counterfactual 1:
#    RUL: 1041.66 (slightly lower than the original 1043.0)
#    Change: A dramatic increase in Charging Time to 474295.42, but no significant changes in other features.
#
# 2. Counterfactual 2:
#    RUL: 1042.92 (closer to the original 1043.0)
#    Change: A significant increase in Time at 4.15V to 79539.08, keeping other feature values largely unchanged.
#
# 3. Counterfactual 3:
#    RUL: 1041.62 (slightly lower than the original 1043.0)
#    Change: An increase in Charging Time to 592961.46 and a small increase in Min Voltage Charge (V) to 3.424.
#
# 4. Counterfactual 4:
#    RUL: 1043.21 (almost the same as the original 1043.0)
#    Change: A dramatic increase in Time Constant Current (s) to 83974.72.
#
# 5. Counterfactual 5:
#    RUL: 1043.57 (slightly higher than the original 1043.0)
#    Change: An increase in Max Voltage Discharge (V) to 4.105, with all other values unchanged.
#

# %% [markdown]
# # Conclusion:
#
# DiCE has provided a set of counterfactuals where key features like Charging Time, Time at 4.15V, and Voltage Discharge have been altered, but the predicted RUL remains quite stable. This suggests that, for the particular instance, small modifications in these features don’t drastically change the remaining useful life predictions.
#

# %% [markdown]
#

# %% [markdown]
#
