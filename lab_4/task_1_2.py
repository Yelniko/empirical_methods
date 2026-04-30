from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLars
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def main():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target

    print(df.isnull().sum())

    s1 = MinMaxScaler()
    s1.fit(df)
    df1 = s1.transform(df)

    df = pd.DataFrame(df1, columns=df.columns)
    print(df)

    #sns.pairplot(df, size=2, hue="MedHouseVal")
    #plt.show()

    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

    corr_target = corr['MedHouseVal'].abs().sort_values(ascending=False)
    top3 = corr_target[1:4].keys()

    for i in top3:
        x = df[[i]]
        y = df['MedHouseVal']
        model1 = LinearRegression()
        model1.fit(x, y)
        print(model1.coef_, model1.intercept_)
        sns.regplot(x=i, y='MedHouseVal', data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        plt.show()

    x = df[top3]
    y = df['MedHouseVal']
    model2 = LinearRegression()
    model2.fit(x, y)
    print(model2.coef_, model2.intercept_)

    # 2

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=0.001),
        "Lasso": Lasso(alpha=0.001),
        "ElasticNet": ElasticNet(alpha=0.001),
    }

    result = []
    predictions = {}

    for model_name, model in models.items():
        model.fit(x, y)
        y_pred = model.predict(x)

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        result.append([model_name, r2, mae, rmse])
        predictions[model_name] = y_pred

    result_df = pd.DataFrame(result, columns=["Model", "R2", "MAE", "RMSE"])

    print(result_df)

    plt.figure(figsize=(12, 10))

    for i, (name, y_pred) in enumerate(predictions.items()):
        plt.subplot(2, 2, i+1)
        sns.scatterplot(x=y, y=y_pred)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

        plt.title(name)
        plt.xlabel("Real")
        plt.ylabel("Predicted")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()