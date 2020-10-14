## Marketing Mix Model (MMM) with sklearn 
An MVP for a MMM in a online store based on the paper [Challenges And Opportunities In Media Mix Modeling - David Chan,Michael Perry](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/2d0395bc7d4d13ddedef54d744ba7748e8ba8dd1.pdf)

## Setup
Excecute: 
```
    conda env create -f env.yml
```

## Algorithm Description

Using sklearn `lineal_model` library, select and fit a list of models. Given a monotonicity criterion over the predictions of each model, the algorithm select some models that qualify with the criterion and create an assembly model.

This assembly model has a unified prediction method that uses CV score to build a "voted" prediction with all qualified models.

This Assembly Model should have a more stable prediction than each individual model.  

![alt text](img.PNG)
