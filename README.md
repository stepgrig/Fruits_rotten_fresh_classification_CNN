# Fruits rotten_fresh Classification
# Data
![1](https://github.com/stepgrig/Fruits_rotten_fresh_classification_CNN/assets/103223897/69d5ed43-4538-4427-bce0-c5267db292f8)
Data landscape of our project aimed at transforming food quality control. We are utilizing the 'Fruits - Fresh and Rotten for Classification' dataset, which is a collection of 13.600 .png images. These images captured by digital cameras serve as a principle of our machine learning model. <br>
To ensure a systematic approach, this rich dataset has been organized into two primary folders - 'Train' and 'Test'. The 'Train' folder, as the name suggests, is used to train our model, and contains a diverse mix of 10,451 .png images. These are further divided across six categories: rotten oranges, rotten bananas, rotten apples, fresh oranges, fresh bananas, and fresh apples. <br>
We have set aside a portion of these images for training and validation purposes. A total of 8,723 images forms the corpus of our training data. This data is used to teach our model the intricate differences between fresh and rotten fruits. Simultaneously, we used 2,178 images for validation purposes. These images serve as validation for the trained model, allowing us to verify its accuracy and fine-tune it for optimal performance. <br>
In essence, this selected and organized data forms the foundation of our project. It equips our model with the 'eye' that separates fresh produce from the rotten and promises quality control processes across food businesses. <br>
# Business Problem
In today's fast-paced world, supermarkets and grocery stores play a crucial role in feeding cities and towns across the globe. However, they face numerous challenges in their daily operations, one of which is ensuring the freshness and quality of the food they sell, especially perishable items like fruits and vegetables.
The classification of food as fresh or rotten is a vital aspect for supermarkets for several reasons:
Customer Satisfaction and Trust: No customer wants to buy rotten fruits or vegetables. Selling fresh, high-quality produce is crucial for maintaining customer satisfaction and trust. If a customer purchases rotten food, it's not only a health risk for the consumer, but it also damages the reputation of the store, which can have long-term negative effects on the business.

Regulatory Compliance: In many jurisdictions, there are strict regulations regarding the sale of fresh food. Selling rotten or spoiled food can lead to hefty fines, lawsuits, and even the closure of the store. Classifying and removing rotten food ensures compliance with health and safety standards.

Reducing Waste: A significant amount of food is wasted in supermarkets due to spoilage. By identifying and removing rotten food quickly, stores can manage their inventories more effectively, reduce waste, and increase their sustainability efforts.

Financial Efficiency: Spoiled food represents a financial loss. By ensuring efficient classification of fresh and rotten produce, supermarkets can improve their bottom lines. The cost of managing and disposing of rotten food can also be reduced.

Accurate classification of fresh and rotten food is key for supermarkets. It helps them to provide better service to their customers, comply with regulations, reduce waste, and enhance financial performance. And as we move towards a more technologically-driven future, leveraging AI and machine learning in this classification process can revolutionize this aspect of supermarket operation.

The use of an AI model to automate the process of distinguishing between fresh and rotten fruits can indeed result in significant cost savings for supermarkets in several ways:

Labor Cost Reduction: Traditionally, checking the freshness of fruits and other perishables is done manually, which is labor-intensive. An AI model can automate this process, reducing the amount of human labor needed and thereby lowering labor costs.

Minimizing Waste: The sooner a supermarket can identify a product that is going bad, the quicker they can take action - perhaps marking it down for quick sale before it becomes unsellable, or using it in prepared foods. This leads to a reduction in waste and associated disposal costs.

Inventory Management: An AI model can help supermarkets manage their inventories more efficiently. With a better understanding of their stock's freshness, they can make more informed decisions about what to order and when, avoiding overstocking perishable items that might not sell before going bad.

Preventing Loss of Sales: Customers dissatisfied with the quality of produce may choose to shop elsewhere, resulting in a loss of sales. By ensuring the freshness of their produce, supermarkets can maintain customer satisfaction, and therefore sales.

Regulatory Fines: Selling rotten fruits can lead to regulatory fines and penalties. An AI model can help supermarkets avoid these costs by ensuring they're compliant with health and safety regulations.

Reduction in Legal Costs: Selling spoiled or contaminated food can also potentially result in lawsuits. Automating the process of identifying and removing rotten fruits can help prevent such legal issues and the associated costs.

Energy Efficiency: By reducing the amount of time that perishable items spend in cold storage, supermarkets could potentially also realize energy savings.

In conclusion, the implementation of AI models for identifying rotten fruits can lead to cost savings in labor, inventory management, waste disposal, and more, improving the overall operational efficiency and profitability of supermarkets.

# Convolutional Neural Network (CNN)
Convolutional Neural Network (CNN) model can play a crucial role in addressing this problem.

The strength of CNNs lies in their ability to extract high-level features from raw input data via multiple layers of convolution and pooling operations, coupled with non-linear activation functions. Convolution layers apply filters to the input data, performing an element-wise product to produce a feature map. The activation function, such as ReLU (Rectified Linear Unit), then applies a non-linear transformation, introducing the capability to learn and represent complex patterns in the data.

Pooling layers further help to reduce the dimensionality, thereby increasing the model's computational efficiency and making it more resistant to overfitting. Specifically, Max pooling, as mentioned, retains only the maximum value from a subset of the input, discarding all other values.

After several iterations of these operations, the model then employs fully connected, or dense, layers to translate the filtered and reduced features into final classification probabilities. The data at this stage is flattened into a one-dimensional array and passed through a Softmax activation function to output the likelihood of each class.

The training process involves fine-tuning the filter values and weights in the neural network to minimize the difference between the model's predictions and the actual labels in the training data. This is achieved through a method called backpropagation, where the model learns from its errors by adjusting its weights in the direction that reduces the loss, using a technique known as gradient descent.
# Model Performance

![1](https://github.com/stepgrig/Fruits_rotten_fresh_classification_CNN/assets/103223897/563d417a-15d2-482e-8371-eb29abbd1c6f)

# Confusion Matrix

This plot will show where the models predictions are correct which is diagonal elements in matrix and will show where the model has mistaken off diagonal elements. Darker the color higher the number of inctances.
![2](https://github.com/stepgrig/Fruits_rotten_fresh_classification_CNN/assets/103223897/4a8c971a-ac9d-4a51-bbef-1822d65e2cf8)

# Classification Report

Precission this is ability of classifier not to label sample positive the is negative. Recall(sensitivity) this is ability of classifier to find all positive samples. F1-score this is weighted mean of precission and recall. Support is number of samples of true response.


![Screenshot (38)](https://github.com/stepgrig/Fruits_rotten_fresh_classification_CNN/assets/103223897/ca38522a-ee9d-4e29-b38e-cbeb1b6bf56f)


# Displaying the images along with its true and predicted labels


![Screenshot (39)](https://github.com/stepgrig/Fruits_rotten_fresh_classification_CNN/assets/103223897/96978e6c-a158-467b-94a8-79e14cdb2578)


By deploying such a CNN model in supermarkets, we can automate the quality control process for fruits, improving efficiency, reducing costs, and enhancing the overall customer experience. Our journey into this project thus far has shown promising results, and we are excited about its potential impact on the supermarket industry.
