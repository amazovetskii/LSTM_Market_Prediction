# Machine Learning techniques in stocks market

Here I am using ML to predict the market data. I try different features and come up with ideas to improve the performance 

You can check out the latest examples in the respective directory

## Key insights and my notes:
<ul>
    <li>
    LSTM showed quite interesting results, considering much fewer parameters in the model. However, vanishing gradient problem can be observed, 
    and model fails to capture long term dependencies.
    </li>
    <li>
    Transformers can perform worse than expected, even a simple lstm can outperform it. 
    In fact vanishing gradient problem can be encountered, especially on encoder part, see transformer_gradients in reports/images
    </li>
</ul>

