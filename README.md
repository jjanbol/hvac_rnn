# Introduction
Heating, ventilation, and air conditioning (HVAC) systems in commercial buildings often operate on static schedules that do not reflect real-time occupancy, leading to substantial energy waste. This project addresses the critical need for dynamic HVAC control by proposing an innovative solution that predicts room occupancy using machine learning. By integrating camera-based occupancy detection with a recurrent neural network (RNN), our approach forecasts occupancy trends to intelligently adjust HVAC settings, ensuring comfortable conditions only during periods of actual use. This predictive method fills a notable gap in current reactive HVAC systems, which struggle to adapt promptly to occupancy fluctuations and incur unnecessary energy costs. Validation through a thermodynamic simulation and real-world data demonstrates the potential for significant energy savings without compromising occupant comfort, offering a promising advancement in sustainable building management.

# Section II: Current progress and concerns
We have successfully setup and sensed using a temperature/humidity sensor (DHT22) with a Raspberry Pi Pico W. We have also successfully published these readings using a MQTT Client server with feeds of data to a dashboard on Adafruit. These processes are not directly related to our project solution but are steps towards building up our ability to implement sensing hardware for this project. We were able to complete this step using at home wifi, but were not able to do so using the Carnegie Mellon IoT wifi connection yet. Our planned implementation, as of now, is for sensing to occur at CMU so we would need to overcome this issue working with the CMU network if we were to utilize publishing data to Adafruit as part of our end solution for this project. 

We have gone through the setup process and initiated data collection using an Infrared PIR Motion Sensor Module (HC-SR501). We have identified a few issues with utilizing this sensor that might lead it to be unsuitable for use during the project. The first issue we have encountered is that after connecting the sensor, there are a significant number of false positives for motion that are detected. We have gone through troubleshooting procedures to include adjusting the sensitivity and timing dials on the PIR sensor, adjusting both the time delay to prevent spamming and the debounce period after a positive reading for motion, and have also tried multiple locations for the sensor to ensure that external interference is not a factor. Another issue that we have identified is that the output of the PIR motion sensor does not appear to provide useful outputs that we could utilize to train a model to make occupancy predictions with. At this time, we have determined to no longer pursue using the Infrared PIR Motion Sensor Module for this project.

Additionally we have tested the YOLO V8n model locally on a laptop that takes a web camera live feed to detect humans in the room. The code needs to be edited and updated further to aggregate the number of people in the room over time and send the gathered data (timestamp, occupancy, etc) to the RNN model that will be used to predict low occupancy within a room over a certain window of time. Currently the model is at 30 FPS and further reduction in FPS might serve better in aggregating the occupancy accurately over time. To accomplish this, we have ordered a Raspberry Pi Mini Camera which will provide the feed that will run on the raspberry device instead of laptop. As a result we will also need to update the current code further to make it suitable with the ordered camera. We expect this camera to arrive no later than April 4th, at which point we will test for suitability. 



# Section III: Plan for Completion of the Project  

Below is a table outlining the plan for project completion, including milestones and dates. Milestones highlighted in gray are those with class due dates or confirmed scheduling with at least one class instructor.  

# **Project Milestone Schedule**  

| Description | Completion Date | Status |
|------------|----------------|--------|
| **Project Proposal** | **2/25/25** | 100% Completed |
| **Receive Initial Sensing Kit** | **3/25/25** | 100% Completed |
| **Project Update Due** | **4/3/25** | 100% Completed |
| **Order and Receive Any Additional Sensors Required** | **4/4/25** | 100% Completed |
| **First Scheduled Group Office Hour/Project Update** | **4/10/25** | 100% Completed |
| **Sensing Hardware Ready for Deployment** | **4/12/25** | 100% Completed |
| **Begin Sensor Deployment/Data Collection** | **4/15/25** | 100% Completed |
| **End Sensor Deployment/Data Collection** | **4/18/25** | 100% Completed |
| **Train Model and Complete Simulation** | **4/21/25** | 100% Completed |
| **Develop Initial Project Demonstration** | **4/21/25** | 100% Completed |
| **Second Scheduled Group Office Hour/Project Update** | **4/22/25** | 100% Completed |
| **Finalize Project Demonstration** | **4/23/25** | 100% Completed |
| **Project Demonstration Event** | **4/25/25** | 100% Completed |
| **Project Report Due** | **5/5/25** | 100% Completed |

