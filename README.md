# AU339_project
1. Introduction

    This is the course project for ***SJTU-AU339-Network Intelligence and Optimization*** . We test the existing patrolling and dispatching algorithm in a gym-based simulation on a guard-and-thief scene.

2. Requirements

    Python 3.7, gym

3. Installation

    We modify the ***gym*** module for some additional function. Concretely, you should add the function below to ***class Geom*** in ***rendering.py***.

    ```python
        def set_color_rgbd(self, r, g, b, d): 
            self._color.vec4 = (r, g, b, d)
    ```

4. To-do list

    - [x] multi guards
    - [x] scalable map
    - [x] thief model
    - [x] alert on theft event and perception of other guards nearby
    - [x] visualize the sight of guards
    - [x] add cartoon images to guards and thieves
    - [x] patrolling algorithm
