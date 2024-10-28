# CNN_Image_retrieval
# A deep learning based approach for image retrieval extraction in mobile edge computing

# Abstract
Deep learning has been widely explored in 5G applications, including computer vision, the Internet of Things (IoT), and intermedia classification. However, applying the deep learning approach in limited-resource mobile devices is one of the most challenging issues. At the same time, users’ experience in terms of Quality of Service (QoS) (e.g., service latency, outcome accuracy, and achievable data rate) performs poorly while interacting with machine learning applications. Mobile edge computing (MEC) has been introduced as a cooperative approach to bring computation resources in proximity to enduser devices to overcome these limitations. This article aims to design a novel image reiterative extraction algorithm based on convolution neural network (CNN) learning and computational task offloading to support machine learning-based mobile applications in resource-limited and uncertain environments. Accordingly, we leverage the framework of image retrieval extraction and introduce three approaches. First, privacy preservation is strict and aims to protect personal data. Second, network traffic reduction. Third, minimizing feature matching time. Our simulation results associated with real-time experiments on a small-scale MEC server have shown the effectiveness of the proposed deep learning-based approach over existing schemes. 

### To Run the code locally do the following :

* Activate the python enviroment:  `source mobile_cnn/bin/activate`
* Specifiy the `IP` address of the server based on the network you have (`ifconfig`). 
* Start the server on the local.host: `~/mobile_cnn/bin/python flask_server.py`
* Start the mobile app (Android client) and select an image.
* Send the image to the server and wait for the result. 
