---
layout: support-page
title: Getting Started with Python
tags: [python]
---

# We found a witch may we burn her?

python has a very intuative (and tollerant) declaration system. Here we declare some variables. A variable is stored in the computers memory for later use.


```python
is_witch = False
number_of_warts = 1
weight_kg = 56.
weight_of_a_duck_kg = 1.6
is_made_of_wood = "do not know"
```

## Hypothesis #1

Here we use some of the variables that we declared in the cell above and print some of the results to the screen. See if you can guess the results!

Probability to be a witch is dependent on the number of warts she has divided by her weight.


```python
print("is she made of wood?", is_made_of_wood)
print("What's, the probability that she's a witch?", number_of_warts/weight_kg)
print("Is she a witch?", is_witch)
print("may we burn her?", is_witch)
```

## Hypothesis #2

A core concept of programming is conditions and logic. A computer scientist goes to the shop and asks her girlfriend what she should buy. "a loaf of bread and if they have eggs buy a dozen", the computer scientist returns with 12 loaves of bread. This is logic. In python we can express this with the `if`, `elif` and `else` conditional statements followed by the `:`.

Maybe we need a combination of witch like traits?


```python
if "no" not in is_made_of_wood:
    is_witch = True
elif number_of_warts > 0:
    if weight_kg < 1.6 or weight_of_a_duck_kg >= 50:
        is_witch = True
else:
    is_witch = False

print("may we burn her?", is_witch)
```

## Hypothesis #3

Variables can be fiddly. Here we demonstrate containers that hold multiple variables for accessing later

Hmmm this is getting complicated. Let's organise things better.


```python
made_of_wood = [1,False,"i don't know"]
witch_traits = {"p(witch)":number_of_warts/weight_kg,
               "wooden":made_of_wood[2],
               "floats":weight_kg < weight_of_a_duck_kg}
```


```python
if witch_traits["p(witch)"] > .7 or "no" not in witch_traits["wooden"] or witch_traits["floats"]:
    is_witch = True
print("may we burn her?", is_witch)
```

## Hypothesis #4

Finally we use all of these tools to answer a question.

How heavy does she have to be to be for us to be allowed to burn her?


```python
weights = [i for i in reversed(range(1,60))]
while not is_witch:
    for weight in weights:
        witch_traits = {"p(witch)":number_of_warts/weight,
                       "wooden":made_of_wood[2],
                       "floats":weight < weight_of_a_duck_kg}
        if witch_traits["p(witch)"] > .7 or "no" not in witch_traits["wooden"] or witch_traits["floats"]:
            is_witch = True
print("we found a {}kg witch. May we burn her? {}".format(weight,is_witch))

```

# Excercise!

Think you've got that? prove it!

[Excercise One - Prime Numbers]({% post_url 2018-06-14-ex01 %})

[Excercise Two - Functions and classes]({% post_url 2018-06-14-ex02 %})



[back to edition]({% post_url 2018-06-14-Getting-Started %})