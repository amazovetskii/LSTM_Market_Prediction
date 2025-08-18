# Stocks prediction using Machine Learning techniques

Here I am using ML to predict the market data. The model predicts 5 new price closes based on the last 30 days 
OHLCV + other features data. I try different features and come up with ideas to improve the performance 

You can check out the latest examples in the respective directory

## Key insights and my notes:
<ul>
    <li>
    Transformers often perform worse than expected, even a simple lstm can outperform it. 
    In fact vanishing gradient problem can be encountered, especially on encoder part, see transformer_gradients in reports/images
    </li>
</ul>


P.S. initially the project was created using pyspark but later the problem of reversing of scaling
arose and decision to switch to pandas was made. But at least i learned PySpark...