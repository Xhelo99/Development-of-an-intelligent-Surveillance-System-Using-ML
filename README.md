# Development of an Intelligent Surveillance System Using ML

## Contents

*   [What is this?](#what-is-this)
*  [Usage](#usage)
*  [Sources](#sources)
        
## What is this?

This paper presents the development of an intelligent surveillance system leveraging machine learning with semantic segmentation to achieve precise object recognition and classification in real-time monitoring scenarios. Semantic segmentation, which 
assigns category labels to every pixel in an image, was utilized to detect objects such as people, vehicles, and buildings within a scene. The system's primary aim is to accurately recognize object classes and track the directional entry of persons within 
a surveillance video feed. The system design comprises three main steps: dataset selection, model training, and testing. We used the PP-LiteSeg model, optimized for semantic segmentation tasks, and trained it on the Cityscapes dataset to ensure accurate performance. 
Additionally, a custom Python-based graphical user interface (GUI) was developed using OpenCV, enabling seamless user interaction and real-time tracking of entry direction. The system sends information to a database regarding a person's duration and entry position, 
facilitating data storage for analysis. Continuous functionality tests validated the system's robustness and reliability, supporting its application in automated surveillance.

## Usage
When the program starts, it can detect up to 20 classes using the [PP-LiteSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs/pp_liteseg) model. If a person is detected, a message is sent to the database with the person's entry side. 
Then, the person is tracked, and a message is sent to the database with the side of the exit.

![Image](https://github.com/Xhelo99/Development-of-an-intelligent-Surveillance-System-Using-ML/blob/master/images/person.png)


[Database image](https://github.com/Xhelo99/Development-of-an-intelligent-Surveillance-System-Using-ML/blob/master/images/db.png)



## Sources
1. [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10)














