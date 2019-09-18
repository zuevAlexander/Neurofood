# Neurofood

### Up
```
docker-compose up -d
```

### Description

The software system is designed to predict user behavior when ordering goods(food in our case) through machine learning.
To create a portrait of the user we need to have the history of his orders(your api is closed for external usage, but you can see test data in folders data and neuro/params/).
In our case we had 300 unique dishes and 75 features for each dish: price, dish category(like soup, salad, garnish, etc.), ingredients, number of orders, etc.
Our system allows to speed up the process of selecting products by the user. Also to increase the number of sales due to offer of new products that fit the profile of the user.


