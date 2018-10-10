a2clstm_cartpole.py  
----------------------------------------------------------------------
A A2C-LSTM algorithm for solving a simple POMDP(partially observed MDP) cart pole problem.  
For a standard full observated cartpole, the state representation is in form of:  
1.standard cartpole Observation:   
        Type: Box(4)  
        Num	Observation                 Min         Max  
        0	Cart Position             -4.8            4.8  
        1	Cart Velocity             -Inf            Inf  
        2	Pole Angle                 -24°           24°  
        3	Pole Velocity At Tip      -Inf            Inf  
Thus I delete Num 1 Cart Velocity attribute, using LSTM to fit the rollout cart position history h(t) for estimating Num 1 Cart Velocity back, as experiment goes, looks worked well.  
2.partially observed cartpole Observation:   
        Type: Box(4)  
        Num	Observation                 Min         Max  
        0	Cart Position             -4.8            4.8  
        1	Pole Angle                 -24°           24°  
        2	Pole Velocity At Tip      -Inf            Inf  

the sample code was written in pytorch, and other algorithms, such as DRQN, Recurrent Policy Gradient can also be implemented like this.  


lstm-train-test.py  
----------------------------------------------------------------------
Is a simple LSTM sequence fitting experimental code, clearly shows how LSTM works.  


All code snippets was created by Haiyinpiao(haiyinpiao@qq.com)  
