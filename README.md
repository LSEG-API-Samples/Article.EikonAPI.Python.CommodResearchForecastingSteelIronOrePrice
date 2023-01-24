# Forecasting steel and iron ore prices using Fbprophet Python library
The purpose of this article is to describe the process used in our model which employs Fbprophet to forecast the price of HRC steel prices in China and iron ore 62% Fe content CFR China prices.

![commod-metal-forecast-02](https://user-images.githubusercontent.com/89068039/211773338-669734d5-28ec-49d1-99d6-9946da2f1744.png)

Example Code Disclaimer:
ALL EXAMPLE CODE IS PROVIDED ON AN “AS IS” AND “AS AVAILABLE” BASIS FOR ILLUSTRATIVE PURPOSES ONLY. REFINITIV MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, AS TO THE OPERATION OF THE EXAMPLE CODE, OR THE INFORMATION, CONTENT, OR MATERIALS USED IN CONNECTION WITH THE EXAMPLE CODE. YOU EXPRESSLY AGREE THAT YOUR USE OF THE EXAMPLE CODE IS AT YOUR SOLE RISK.

## Table of contents
* [Introduction](#introduction)
* [Prerequisite](#prerequisite)
* [Hypothesis](#hypothesis)
* [Conclusion](#conclusion)
* [Authors](#author)
* [References](#references)

## <a id="introduction"></a>Introduction
China’s steel market has been very volatile in the past several years. It is inherently difficult to forecast commodities prices, particularly in the post-pandemic environment but in the case of steel and iron ore (the main material used in steel production), this task is even harder for a few reasons.

Firstly, China’s government has embarked on a de-carbonization journey which resulted in steel production cuts and capacity replacements, particularly in the highly polluted industrial areas of Tangshan. Since January 2021, there have been many announcements from the government and official media confirming China’s plans to become carbon neutral by 2060. Some of the announced measures were addressed directly at the steel industry which has already undergone major transformations including mergers and acquisitions and other structural adjustments like shutting down outdated capacities and plans to introduce green technology like hydrogen and carbon capturing.

Secondly, China has been struggling with some structural challenges in the property market. A combination of zero-covid rules, a slowdown in real estate, and environmental production cuts have been hurting steel demand for the better part of last year. However, last month China’s government announced 16 measures to revive the housing market. The measures among other things aim to provide financial support for viable construction developments, resume stalled projects and help mortgage holders. Although the new property policies do a lot to address the problem, they may not help solve the main issue of the little trust China’s buyers have in real-estate developers.

Thirdly, protectionists policies in various countries across the world are becoming a more prominent feature of the governments’ economic plans. This results in high tariffs imposed on steel products from China. Lastly, the tragic events in Ukraine do not only cost lives in the region and energy crisis in Europe but also threaten economic stability in the rest of the world by reshaping supply chains and causing a major supply crisis.

All the above must be considered in qualitative price forecasting as these developments define fundamentals. One might argue that in such an environment qualitative approach should be the only way. However, on the other hand, predicting the outcome of these uncertainties is extremely difficult and time series forecasting could still be useful in certain circumstances.

## <a id="prerequisite"></a>Prerequisite 
To run the code in this article, any of the below can be used

- The [CodeBook](https://www.refinitiv.com/en/products/codebook) application inside the Eikon Desktop/Refinitiv Workspace
  - The code is available in the CodeBook application (Example folder) ****[TO DO]**** **add direct URL to codebook example**
  - The codeBook is a feature that provides you with a cloud-hosted development environment for Python scripting, and supports bespoke workflow design. With CodeBook, we are redefining how you access, evaluate and build on the breadth and depth of our data. Whether you are a professional developer, occasional coder or have no coding skills, CodeBook offers the capabilities for you to design your own data models or access Python applications built by your colleagues – the possibilities are endless. Available in Refinitiv Workspace and Eikon, CodeBook also gives you access to our APIs and platform services in a single interface so you can be even more efficient. With our data, you can build analytics, applications and other use cases that are critical for your daily workflow needs. The Jupyter notebook file is located in _/__Examples__/06. Commodities Research/_ (screenshot below)
- Or your own local Python Environment with
  - Required Python libraries and their version:
    - eikon==1.1.14
      - [Eikon Data API Quick-start guide](https://developers.refinitiv.com/en/api-catalog/eikon/eikon-data-api/quick-start).
    - fbprophet==0.7.1
    - statsmodels==0.11.0
    - numpy==1.22.4
    - pandas==1.3.5
    - datetime==4.3
    - plotly==5.4.0
  - Jupyter Lab or Jupyter Notebook application is installed and ready to use

## <a id="hypothesis"></a>Hypothesis
It has been written a lot on the topic of time series forecasting. The purpose of this article is to describe the process used in our model which employs [fbprophet](https://facebook.github.io/prophet/) Python library (Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.) to forecast the price (of HRC steel prices in China and iron ore 62% Fe content CFR China prices, in this article)

There are multiple reasons why the Fbprophet Python library could be a good fit for our forecast modeling:

1. It works well with stationary time series where non-linear trends follow some type of seasonality (weekly, yearly, daily).
2. Fbprophet requires several cycles of such seasonality to be observed in the historical time series
3. Fbprophet is relatively robust with shifts in data trends and deals well with outliers.

## <a id="Step-by-Step Process"></a>Step-by-Step Process
To start with we load RICs and prepare data for further analysis.

Secondly, we perform [Augmented Dickey-Fuller Test](https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/) (ADF) to check whether our data is stationary/non-stationary. Using the *diff()* function will allow us to calculate the first-order difference in the selected time series with a predefined shift period. The shift period is defined as a parameter *period*. We then test the resulting row of data for stationarity/non-stationarity and print the test statistics as per the method below.

After performing the stationarity check, we need to define the model’s characteristics and “tune it” so to speak. We define the prediction period as a *period* of 30 days (this could be increased). We then create *ds* (date as DateTime) and *y* (values that we are going to predict) columns. We will then convert the Price column into a log value. In statistical analysis, logging helps to convert multiplicative relationships between datapoints into additive relationships, and converts exponential (compound growth) trends into linear trends. Therefore, by taking logarithms of values which are multiplicatively related we can explain their behaviour with a linear model.

We are taking data (except for the last 30 values) for training our model in order to compare the actual values in our time series with the 30 values predicted by the model

```ric_train = ric[:-period]```

We create the model, using the **fit** method, and get the prognosis. It is worth mentioning that Fbprophet is a very flexible and easy-to-train model. Very often the quality of the forecast depends on the time series' characteristics as well as the model’s parameters that can be defined (and tuned) depending on our knowledge about the time series and multiple rounds of testing. These parameters include:

- **Interval_width:** the width of the uncertainty intervals
- **N_changepoints:** changepoints are the DateTime points where the time series have sudden changes in the trajectory (there is a full list of changepoints in prophet which are accessible using ch_points = pr_model.changepoints)
- **Changepoints_prior_scale:** helps adjust the strength of the trend. It is possible to increase the value of changepoints_prior_scale to make the trend more flexible
- **Changepoints_range:** the range of historical data points (starting from the first observation) within which the changepoints in trend will be assessed
- **Yearly, weekly and daily seasonality:** self-explanatory parameters that we can set as true or false for our time series
- **Holidays:** we can create a custom holiday list by creating a data frame with two columns ds and holiday
For predictions, we need to create a data frame with ds (DateTime) containing the dates for which we want to create a forecast. We are using make_future_dataframe() where we specify prediction dates. We then combine actual data and our forecast data into one table. It is useful to have a look at **MAE (mean absolute error)** which helps in understanding how precise the model’s predictions are.

Lastly, we perform the prediction of all available data points within our time series and as a final step, we are plotting our data and predictions using the Plotly Python library

## <a id="conclusion"></a>Conclusion
This article demonstrates how the Fbprophet Python Library can be used in price forecast modelling using an example of Hot Rolled Coil (HRC) steel price in China and iron ore 62% Fe content CFR China. The prices’ historical data and predictions are plotted as a graph with Plotly Python library as a visualisation to help with analysing price trends and a potential direction the prices may take in the short-term future.

## <a id="author"></a>Authors
- Thorne, Tamara (tamara.thorne@lseg.com)
- Refinitiv Developer Advocate (https://developers.refinitiv.com/en)

## <a id="references"></a>References
For further details, please check out the following resources:
* [CodeBook application](https://www.refinitiv.com/en/products/codebook)
* [Refinitiv Eikon Data API: Quick Start](https://developers.refinitiv.com/en/api-catalog/eikon/eikon-data-api/quick-start). 
* [FB Prophet Python library](https://facebook.github.io/prophet/).
* [Augmented Dickey Fuller Test (ADF Test) – Must Read Guide](https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/).

For any question related to the Eikon Data API, please use the Developers Community [Q&A Forum](https://community.developers.refinitiv.com/spaces/92/index.html).
